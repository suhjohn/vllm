import argparse
import os

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid


def concurrency_controller() -> bool:
    # Compute pending sequences
    total_pending_sequences = len(
        engine.engine.scheduler.waiting) + len(engine.engine.scheduler.swapped)
    return total_pending_sequences > os.environ.get('MAX_PENDING_SEQUENCES', 0)


async def handle_stream(job):
    request_dict = job["input"]
    prompt = request_dict.pop("prompt")
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()
    results_generator = engine.generate(prompt, sampling_params, request_id)
    previous_texts = [""] * sampling_params.n
    async for res in results_generator:
        for output in res.outputs:
            i = output.index
            delta_text = output.text[len(previous_texts[i]):]
            previous_texts[i] = output.text
            yield delta_text.encode("utf-8")


async def handle(job):
    request_dict = job["input"]
    prompt = request_dict.pop("prompt")
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()
    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    assert final_output is not None

    text_outputs = [output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    return ret


if __name__ == "__main__":
    import runpod
    parser = argparse.ArgumentParser()
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    is_streaming = os.environ.get('STREAMING', False) == 'True'
    if is_streaming:
        print("Starting the vLLM serverless worker with streaming enabled.")
        runpod.serverless.start(
            {"handler": handle_stream, "concurrency_controller": concurrency_controller})
    else:
        print("Starting the vLLM serverless worker with streaming disabled.")
        runpod.serverless.start(
            {"handler": handle, "concurrency_controller": concurrency_controller})
