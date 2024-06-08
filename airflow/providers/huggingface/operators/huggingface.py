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

from functools import cached_property
from typing import TYPE_CHECKING, Any

from airflow.models import BaseOperator
from airflow.providers.huggingface.hooks.huggingface import HuggingFaceHook

if TYPE_CHECKING:
    import numpy as np
    from huggingface_hub.inference._types import ConversationalOutput

    from airflow.utils.context import Context


class HuggingFaceSpeechRecognitionOperator(BaseOperator):
    """
    Perform automatic speech recognition (ASR or audio-to-text) on the given audio content.

    :param conn_id (`str`): The connection ID to use when connecting to Huggingface
    :param model (`str`, *optional*):
        The model to use for ASR. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
        Inference Endpoint. If not provided, the default recommended model for ASR will be used.
    :param audio (Union[str, Path, bytes, BinaryIO]):
        The content to transcribe. It can be raw audio bytes, local audio file, or a URL to an audio file.
    :param inference_params (`Dict`, *optional*):
        Additional parameters to pass to the Inference API. See the documentation for more details.
    """

    template_fields = ("model", "audio")

    def __init__(
        self,
        *,
        conn_id: str,
        model: str | None = None,
        audio: str | bytes | None = None,
        inference_params: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.conn_id = conn_id
        self.model = model
        self.audio = audio
        self.inference_params = inference_params or {}

    @cached_property
    def hook(self) -> HuggingFaceHook:
        """Instantiate and cache the hook."""
        return HuggingFaceHook(conn_id=self.conn_id)

    def execute(self, context: Context) -> str:
        """Execute the operator."""
        client = self.hook.inference(**self.inference_params)
        return client.automatic_speech_recognition(
            model=self.model,
            audio=self.audio,
        )


class HuggingFaceEmbeddingsOperator(BaseOperator):
    """
    Perform embeddings on the given text content.

    :param conn_id (`str`): The connection ID to use when connecting to Huggingface
    :param model (`str`, *optional*):
        The model to use for embeddings. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
        Inference Endpoint. If not provided, the default recommended model for embeddings will be used.
    :param text (str):
        The text to embed.
    :param inference_params (`Dict`, *optional*):
        Additional parameters to pass to the Inference API. See the documentation for more details.
    """

    template_fields = ("model", "text")

    def __init__(
        self,
        *,
        conn_id: str,
        model: str | None = None,
        text: str | None = None,
        inference_params: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.conn_id = conn_id
        self.model = model
        self.text = text
        self.inference_params = inference_params or {}

    @cached_property
    def hook(self) -> HuggingFaceHook:
        """Instantiate and cache the hook."""
        return HuggingFaceHook(conn_id=self.conn_id)

    def execute(self, context: Context) -> np.ndarray:
        """Execute the operator."""
        client = self.hook.inference(**self.inference_params)
        return client.feature_extraction(
            model=self.model,
            text=self.text,
        )


class HuggingFaceQuestionAnsweringOperator(BaseOperator):
    """
    Perform question answering on the given text content.

    :param conn_id (`str`): The connection ID to use when connecting to Huggingface
    :param model (`str`, *optional*):
        The model to use for question answering. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
        Inference Endpoint. If not provided, the default recommended model for question answering will be used.
    :param question (str):
        The question to answer.
    :param context (str):
        The context to search for the answer.
    :param inference_params (`Dict`, *optional*):
        Additional parameters to pass to the Inference API. See the documentation for more details.
    """

    template_fields = ("model", "question", "context")

    def __init__(
        self,
        *,
        conn_id: str,
        model: str | None = None,
        question: str | None = None,
        context: str | None = None,
        inference_params: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.conn_id = conn_id
        self.model = model
        self.question = question
        self.context = context
        self.inference_params = inference_params or {}

    @cached_property
    def hook(self) -> HuggingFaceHook:
        """Instantiate and cache the hook."""
        return HuggingFaceHook(conn_id=self.conn_id)

    def execute(self, context: Context) -> str:
        """Execute the operator."""
        client = self.hook.inference(**self.inference_params)
        return client.question_answering(
            model=self.model,
            question=self.question,
            context=self.context,
        )


class HuggingFaceSummarizationOperator(BaseOperator):
    """
    Perform summarization on the given text content.

    :param conn_id (`str`): The connection ID to use when connecting to Huggingface
    :param model (`str`, *optional*):
        The model to use for summarization. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
        Inference Endpoint. If not provided, the default recommended model for summarization will be used.
    :param text (str):
        The text to summarize.
    :param inference_params (`Dict`, *optional*):
        Additional parameters to pass to the Inference API. See the documentation for more details.
    """

    template_fields = ("model", "text")

    def __init__(
        self,
        *,
        conn_id: str,
        model: str | None = None,
        text: str | None = None,
        inference_params: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.conn_id = conn_id
        self.model = model
        self.text = text
        self.inference_params = inference_params or {}

    @cached_property
    def hook(self) -> HuggingFaceHook:
        """Instantiate and cache the hook."""
        return HuggingFaceHook(conn_id=self.conn_id)

    def execute(self, context: Context) -> str:
        """Execute the operator."""
        client = self.hook.inference(**self.inference_params)
        return client.summarization(
            model=self.model,
            text=self.text,
        )


class HuggingFaceTextGenerationOperator(BaseOperator):
    """
    Given a prompt, generate text using the given model.

    :param conn_id (`str`): The connection ID to use when connecting to Huggingface
    :param model (`str`, *optional*):
        The model to use for text generation. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
        Inference Endpoint. If not provided, the default recommended model for text generation will be used.
    :param prompt (`str`)`:
        The prompt to generate text from.
    :param details (`bool`):
        By default, text_generation returns a string. Pass `details=True` if you want a detailed output (tokens,
        probabilities, seed, finish reason, etc.). Only available for models running on with the
        `text-generation-inference` backend.
    :param stream (`bool`):
        By default, text_generation returns a string. Pass `stream=True` if you want a stream of text. Only available
        for models running on with the `text-generation-inference` backend.
    :param do_sample (`bool`):
        Activate logits sampling.
    :param max_new_tokens (`int`):
        Maximum number of tokens to generate.
    :param best_of (`int`):
        Generate best_of sequences and return the one if the highest token logprobs.
    :param repetition_penalty (`float`):
        The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no penalty.
    :param return_full_text (`bool`):
        Whether to prepend the prompt to the generated text.
    :param seed (`int`):
        Random sampling seed
    :param stop_sequences (`List[str]`):
        Stop generating tokens if a member of `stop_sequences` is generated.
    :param temperature (`float`):
        The value used to module the logits distribution.
    :param top_k (`int`):
        The number of highest probability vocabulary tokens to keep for top-k-filtering.
    :param top_p (`float`):
        If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
        higher are kept for generation.
    :param truncate (`int`):
        Truncate inputs tokens to the given size.
    :param typical_p (`float`):
        Typical Decoding mass
        See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information.
    :param watermark (`str`):
        Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
    :param decoder_input_details (`bool`):
        Return the decoder input token logprobs and ids. You must set `details=True` as well for it to be taken
        into account. Defaults to `False`.
    :param inference_params (`Dict`, *optional*):
        Additional parameters to pass to the Inference API. See the documentation for more details.
    """

    template_fields = ("model", "prompt")

    def __init__(
        self,
        *,
        conn_id: str,
        model: str | None = None,
        prompt: str | None = None,
        details: bool = False,
        stream: bool = False,
        do_sample: bool = False,
        max_new_tokens: int = 20,
        best_of: int | None = None,
        repetition_penalty: float | None = None,
        return_full_text: bool = False,
        seed: int | None = None,
        stop_sequences: list[str] | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        truncate: int | None = None,
        typical_p: float | None = None,
        watermark: bool = False,
        decoder_input_details: bool = False,
        inference_params: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.conn_id = conn_id
        self.model = model
        self.prompt = prompt
        self.details = details
        self.stream = stream
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.best_of = best_of
        self.repetition_penalty = repetition_penalty
        self.return_full_text = return_full_text
        self.seed = seed
        self.stop_sequences = stop_sequences or []
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.truncate = truncate
        self.typical_p = typical_p
        self.watermark = watermark
        self.decoder_input_details = decoder_input_details
        self.inference_params = inference_params or {}

    @cached_property
    def hook(self) -> HuggingFaceHook:
        """Instantiate and cache the hook."""
        return HuggingFaceHook(conn_id=self.conn_id)

    def execute(self, context: Context) -> str:
        """Execute the operator."""
        client = self.hook.inference(**self.inference_params)
        return client.text_generation(
            model=self.model,
            prompt=self.prompt,
            details=self.details,
            stream=self.stream,
            do_sample=self.do_sample,
            max_new_tokens=self.max_new_tokens,
            best_of=self.best_of,
            repetition_penalty=self.repetition_penalty,
            return_full_text=self.return_full_text,
            seed=self.seed,
            stop_sequences=self.stop_sequences,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            truncate=self.truncate,
            typical_p=self.typical_p,
            watermark=self.watermark,
            decoder_input_details=self.decoder_input_details,
        )


class HuggingFaceConversationalOperator(BaseOperator):
    """
    Perform conversational inference on the given text content.

    :param conn_id (`str`): The connection ID to use when connecting to Huggingface
    :param model (`str`, *optional*):
        The model to use for conversational inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a
        deployed Inference Endpoint. If not provided, the default recommended model for conversational inference will
        be used.
    :param text (str):
        TThe last input from the user in the conversation.
    :param generated_responses (`List[str]`, *optional*):
        A list of strings corresponding to the earlier replies from the model. Defaults to None.
    :param past_user_inputs (`List[str]`, *optional*):
        A list of strings corresponding to the earlier replies from the user. Should be the same length as
        `generated_responses`. Defaults to None.
    :param con_parameters (`Dict[str, Any]`, *optional*):
        Additional parameters for the conversational task. Defaults to None. For more details about the available
        parameters, please refer to [this page](https://huggingface.co/docs/api-inference/detailed_parameters#conversational-task)
    :param inference_params (`Dict`, *optional*):
        Additional parameters to pass to the Inference API. See the documentation for more details.
    """

    template_fields = ("model", "text")

    def __init__(
        self,
        *,
        conn_id: str,
        model: str | None = None,
        text: str | None = None,
        inference_params: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.conn_id = conn_id
        self.model = model
        self.text = text
        self.inference_params = inference_params or {}

    @cached_property
    def hook(self) -> HuggingFaceHook:
        """Instantiate and cache the hook."""
        return HuggingFaceHook(conn_id=self.conn_id)

    def execute(self, context: Context) -> ConversationalOutput:
        """Execute the operator."""
        client = self.hook.inference(**self.inference_params)
        return client.conversational(
            model=self.model,
            text=self.text,
        )
