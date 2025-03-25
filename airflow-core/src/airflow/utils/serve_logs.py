# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Serve logs process."""

from __future__ import annotations

import logging
import os
import socket
from collections import namedtuple

import gunicorn.app.base
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from jwt.exceptions import (
    ExpiredSignatureError,
    ImmatureSignatureError,
    InvalidAudienceError,
    InvalidIssuedAtError,
    InvalidSignatureError,
)
from setproctitle import setproctitle
from starlette.status import HTTP_403_FORBIDDEN

from airflow.api_fastapi.auth.tokens import JWTValidator, get_signing_key
from airflow.configuration import conf
from airflow.utils.docs import get_docs_url
from airflow.utils.module_loading import import_string

logger = logging.getLogger(__name__)


def create_app():
    fastapi_app = FastAPI(title="Airflow Logs Server")
    leeway = conf.getint("webserver", "log_request_clock_grace", fallback=30)
    log_directory = os.path.expanduser(conf.get("logging", "BASE_LOG_FOLDER"))
    log_config_class = conf.get("logging", "logging_config_class")
    if log_config_class:
        logger.info("Detected user-defined logging config. Attempting to load %s", log_config_class)
        try:
            logging_config = import_string(log_config_class)
            try:
                base_log_folder = logging_config["handlers"]["task"]["base_log_folder"]
            except KeyError:
                base_log_folder = None
            if base_log_folder is not None:
                log_directory = base_log_folder
                logger.info(
                    "Successfully imported user-defined logging config. FastAPI App will serve log from %s",
                    log_directory,
                )
            else:
                logger.warning(
                    "User-defined logging config does not specify 'base_log_folder'. "
                    "FastAPI App will use default log directory %s",
                    base_log_folder,
                )
        except Exception as e:
            raise ImportError(f"Unable to load {log_config_class} due to error: {e}")
    signer = JWTValidator(
        issuer=None,
        secret_key=get_signing_key("webserver", "secret_key"),
        leeway=leeway,
        audience="task-instance-logs",
    )

    async def validate_token(request: Request):
        try:
            auth = request.headers.get("Authorization")
            if auth is None:
                logger.warning("The Authorization header is missing: %s.", request.headers)
                raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Authorization header is missing")

            payload = signer.validated_claims(auth)
            token_filename = payload.get("filename")
            request_filename = request.path_params.get("filename")

            if token_filename is None:
                logger.warning("The payload does not contain 'filename' key: %s.", payload)
                raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Token payload missing filename")

            if token_filename != request_filename:
                logger.warning(
                    "The payload log_relative_path key is different than the one in token:"
                    "Request path: %s. Token path: %s.",
                    request_filename,
                    token_filename,
                )
                raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Token filename mismatch")

            return payload
        except InvalidAudienceError:
            logger.warning("Invalid audience for the request", exc_info=True)
            raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid audience")
        except InvalidSignatureError:
            logger.warning("The signature of the request was wrong", exc_info=True)
            raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid signature")
        except ImmatureSignatureError:
            logger.warning("The signature of the request was sent from the future", exc_info=True)
            raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Immature signature")
        except ExpiredSignatureError:
            logger.warning(
                "The signature of the request has expired. Make sure that all components "
                "in your system have synchronized clocks. "
                "See more at %s",
                get_docs_url("configurations-ref.html#secret-key"),
                exc_info=True,
            )
            raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Expired signature")
        except InvalidIssuedAtError:
            logger.warning(
                "The request was issues in the future. Make sure that all components "
                "in your system have synchronized clocks. "
                "See more at %s",
                get_docs_url("configurations-ref.html#secret-key"),
                exc_info=True,
            )
            raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid issued at time")
        except Exception:
            logger.warning("Unknown error", exc_info=True)
            raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Unknown error")

    @fastapi_app.get("/log/{filename:path}")
    async def serve_logs_view(filename: str, token_payload: dict = Depends(validate_token)):
        file_path = os.path.join(log_directory, filename)
        if not os.path.isfile(file_path):
            raise HTTPException(status_code=404, detail="Log file not found")
        return FileResponse(
            path=file_path, media_type="application/json", filename=os.path.basename(filename)
        )

    return fastapi_app


GunicornOption = namedtuple("GunicornOption", ["key", "value"])


class StandaloneGunicornApplication(gunicorn.app.base.BaseApplication):
    """
    Standalone Gunicorn application/serve for usage with any WSGI-application.

    Code inspired by an example from the Gunicorn documentation.
    https://github.com/benoitc/gunicorn/blob/cf55d2cec277f220ebd605989ce78ad1bb553c46/examples/standalone_app.py

    For details, about standalone gunicorn application, see:
    https://docs.gunicorn.org/en/stable/custom.html
    """

    def __init__(self, app, options=None):
        self.options = options or []
        self.application = app
        super().__init__()

    def load_config(self):
        for option in self.options:
            self.cfg.set(option.key.lower(), option.value)

    def load(self):
        return self.application


def serve_logs(port=None):
    """Serve logs generated by Worker."""
    setproctitle("airflow serve-logs")
    app = create_app()

    # Create ASGI app with uvicorn

    port = port or conf.getint("logging", "WORKER_LOG_SERVER_PORT")

    # If dual stack is available and IPV6_V6ONLY is not enabled on the socket
    # then when IPV6 is bound to it will also bind to IPV4 automatically
    if getattr(socket, "has_dualstack_ipv6", lambda: False)():
        host = "::"
    else:
        host = "0.0.0.0"

    # Use Gunicorn with Uvicorn workers
    options = [
        GunicornOption("bind", f"{host}:{port}"),
        GunicornOption("workers", 2),
        GunicornOption("worker_class", "uvicorn.workers.UvicornWorker"),
    ]
    StandaloneGunicornApplication(app, options).run()


if __name__ == "__main__":
    serve_logs()
