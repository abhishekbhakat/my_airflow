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

from __future__ import annotations

from typing import Any

from huggingface_hub import InferenceClient
from huggingface_hub.hf_api import HfApi

from airflow.exceptions import AirflowException
from airflow.hooks.base import BaseHook


class HuggingFaceHook(BaseHook):
    """
    Interact with HuggingFace Hub.

    :param token: User access token to generate from https://huggingface.co/settings/token`
    """

    conn_name_attr = "conn_id"
    default_conn_name = "huggingface_default"
    conn_type = "huggingface"
    hook_name = "HuggingFace"

    @classmethod
    def get_ui_field_behaviour(cls) -> dict[str, Any]:
        """Return custom field behaviour."""
        return {
            "hidden_fields": ["schema", "port", "extra", "login", "host"],
            "relabeling": {
                "password": "Token",
            },
        }

    def __init__(
        self,
        conn_id: str = default_conn_name,
        token: str | None = None,
    ) -> None:
        super().__init__()
        self.conn_id = conn_id
        self.token = self.__get_token(token, conn_id)
        self.connection = self.get_conn()

    def __get_token(self, token: str | None, conn_id: str) -> str:
        if token is None:
            conn = self.get_connection(conn_id)

            if not conn.password:
                raise AirflowException("Missing token(password) in HuggingFace connection")

            token = conn.password
        return token

    def get_conn(self) -> HfApi:
        """
        Get an HfApi client.

        :return: HfApi client
        """
        return HfApi(token=self.token)

    def whoami(self) -> dict[str, Any]:
        """
        Get user information.

        :return: User information
        """
        return self.connection.whoami()

    def inference(self,
        model: str | None = None,
        timeout: float | None = None,
        headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
    ) -> InferenceClient:
        """
        Initialize a new Inference Client.

        [`InferenceClient`] aims to provide a unified experience to perform inference. The client can be used
        seamlessly with either the (free) Inference API or self-hosted Inference Endpoints.

        Args:
            model (`str`, `optional`):
                The model to run inference with. Can be a model id hosted on the Hugging Face Hub, e.g. `bigcode/starcoder`
                or a URL to a deployed Inference Endpoint. Defaults to None, in which case a recommended model is
                automatically selected for the task.
            token (`str`, *optional*):
                Hugging Face token. Will default to the locally saved token. Pass `token=False` if you don't want to send
                your token to the server.
            timeout (`float`, `optional`):
                The maximum number of seconds to wait for a response from the server. Loading a new model in Inference
                API can take up to several minutes. Defaults to None, meaning it will loop until the server is available.
            headers (`Dict[str, str]`, `optional`):
                Additional headers to send to the server. By default only the authorization and user-agent headers are sent.
                Values in this dictionary will override the default values.
            cookies (`Dict[str, str]`, `optional`):
                Additional cookies to send to the server.
        """
        return InferenceClient(
            model=model,
            token=self.token,
            timeout=timeout,
            headers=headers,
            cookies=cookies,
        )
