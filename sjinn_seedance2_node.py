import asyncio
import inspect
import io
import os
import time
import wave
import requests
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image as PilImage

try:
    from comfy.utils import ProgressBar as _ComfyProgressBar
    _HAS_PROGRESS_BAR = True
except ImportError:
    _HAS_PROGRESS_BAR = False

try:
    from comfy_api.latest._input_impl.video_types import VideoFromFile as _VideoFromFile
    _HAS_VIDEO_FROM_FILE = True
except ImportError:
    _HAS_VIDEO_FROM_FILE = False

import folder_paths

try:
    from .sjinn_config import SJINN_API_KEY as _DEFAULT_API_KEY, \
        SJINN_SESSION_TOKEN as _DEFAULT_SESSION_TOKEN
except ImportError:
    _DEFAULT_API_KEY = ""
    _DEFAULT_SESSION_TOKEN = ""

# Module-level thread pool — shared across all node instances and async tasks.
# Used for all blocking I/O (uploads, HTTP requests, file ops) so the asyncio
# event loop is never blocked and multiple nodes can run in parallel.
_POOL = ThreadPoolExecutor(max_workers=20, thread_name_prefix="sjinn")


class SjinnSeedance2Node:
    """ComfyUI node: Sjinn.ai Seedance 2.0 video generation.

    Supports up to 8 reference images, 3 reference videos, 2 reference audio clips.
    Downloads result and returns native VIDEO + first_frame + last_frame + frames.

    Async implementation: when multiple independent SjinnSeedance2Node instances exist
    in the same workflow, ComfyUI schedules them as concurrent asyncio tasks — all
    submissions reach the Sjinn API simultaneously without waiting for each other.
    """

    RATIOS      = ["16:9", "9:16", "1:1", "4:3", "3:4"]
    DURATIONS   = ["4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
    SPEED_MODES = ["pro", "fast"]

    BASE_URL   = "https://sjinn.ai"
    UPLOAD_URL = f"{BASE_URL}/api/upload_file"
    CREATE_URL = f"{BASE_URL}/api/create_seedance20_video"
    STATUS_URL = f"{BASE_URL}/api/un-api/query_tool_task_status"

    # ── Timing constants ──────────────────────────────────────────────────────
    # Deadline counts from task *creation* (after all uploads), not from generate() start.
    MAX_POLL_MINUTES    = 30   # was 15 — generous budget for long generations
    POLL_INTERVAL_SEC   = 15   # seconds between status checks
    POLL_RETRY_LIMIT    = 4    # retries on transient network errors before giving up
    POLL_RETRY_DELAY    = 8    # seconds between retries on network error

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session_token": ("STRING", {
                    "default": _DEFAULT_SESSION_TOKEN,
                    "tooltip": "NextAuth session cookie from sjinn.ai browser login. Valid ~30 days.",
                }),
                "api_key": ("STRING", {
                    "default": _DEFAULT_API_KEY,
                    "tooltip": "Sjinn.ai API key — used for status polling only.",
                }),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                # Reference images — mention @Image1 ... @Image8 in prompt
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
                # Reference videos — connect Load Video, mention @Video1 @Video2 @Video3
                # NOTE: Seedance requires reference video ≤ 720P and ≥ 2 seconds
                "video_1": ("VIDEO", {"tooltip": "Connect Load Video node. Use @Video1 in prompt."}),
                "video_2": ("VIDEO", {"tooltip": "Connect Load Video node. Use @Video2 in prompt."}),
                "video_3": ("VIDEO", {"tooltip": "Connect Load Video node. Use @Video3 in prompt."}),
                # Reference audio — connect Load Audio, mention @Audio1 @Audio2
                "audio_1": ("AUDIO", {"tooltip": "Reference audio clip. Use @Audio1 in prompt."}),
                "audio_2": ("AUDIO", {"tooltip": "Reference audio clip. Use @Audio2 in prompt."}),
                # Generation settings
                "ratio":                (cls.RATIOS,      {"default": "16:9"}),
                "duration":             (cls.DURATIONS,   {"default": "5"}),
                "speed_mode":           (cls.SPEED_MODES, {"default": "pro",
                                         "tooltip": "pro = higher quality / slower, fast = lower quality / quicker"}),
                "accelerate_real_face": ("BOOLEAN", {"default": True,
                                         "tooltip": "Preserve face consistency across frames."}),
                "smart_rescue":         ("BOOLEAN", {"default": False,
                                         "tooltip": "Auto-rewrite prompt if it violates content policy. "
                                                    "Disable to use prompt verbatim. "
                                                    "Note: hidden in Sjinn UI but still accepted by the API."}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**32 - 1,
                                 "tooltip": "-1 = random seed, 0–4294967295 = fixed seed"}),
            },
        }

    RETURN_TYPES  = ("VIDEO", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES  = ("video", "first_frame", "last_frame", "frames")
    FUNCTION      = "generate"
    CATEGORY      = "Sjinn.ai"

    # ------------------------------------------------------------------
    # Progress bar helper (safe to call from async context on main loop)
    # ------------------------------------------------------------------
    class _PBar:
        def __init__(self, total: int):
            self._cb  = _ComfyProgressBar(total) if _HAS_PROGRESS_BAR else None
            self._total = total
            self._cur   = 0

        def set(self, value: int):
            value = max(self._cur, min(value, self._total))
            self._cur = value
            if self._cb is not None:
                self._cb.update_absolute(value, self._total)

        def advance(self, delta: int = 1):
            self.set(self._cur + delta)

    # ------------------------------------------------------------------
    # Main entry point — async so ComfyUI runs independent nodes in parallel
    # ------------------------------------------------------------------

    async def generate(
        self,
        session_token: str,
        api_key: str,
        prompt: str,
        image_1=None, image_2=None, image_3=None, image_4=None,
        image_5=None, image_6=None, image_7=None, image_8=None,
        video_1=None, video_2=None, video_3=None,
        audio_1=None, audio_2=None,
        ratio: str    = "16:9",
        duration: str = "5",
        speed_mode: str = "pro",
        accelerate_real_face: bool = True,
        smart_rescue: bool = False,
        seed: int = -1,
    ):
        if not session_token.strip():
            raise ValueError("SjinnSeedance2Node: session_token is required")
        if not api_key.strip():
            raise ValueError("SjinnSeedance2Node: api_key is required")

        images        = [x for x in [image_1, image_2, image_3, image_4,
                                      image_5, image_6, image_7, image_8] if x is not None]
        video_entries = [x for x in [video_1, video_2, video_3] if x is not None]
        audio_entries = [x for x in [audio_1, audio_2] if x is not None]

        print(
            f"[Seedance2] Starting — {len(images)} images, {len(video_entries)} videos, "
            f"{len(audio_entries)} audio | ratio={ratio}, duration={duration}s, "
            f"mode={speed_mode}, face={accelerate_real_face}, rescue={smart_rescue}, seed={seed}"
        )

        generate_start = time.time()
        pbar = self._PBar(100)
        loop = asyncio.get_running_loop()

        # ── Upload all media in parallel ───────────────────────────────
        # Images, videos, and audio are uploaded concurrently — significant speedup
        # when there are multiple reference files.
        # Progress: 0 → 38

        async def _upload_image_async(idx: int, tensor) -> str:
            print(f"[Seedance2] Uploading image {idx}/{len(images)}...")
            try:
                fname = await loop.run_in_executor(_POOL, self._upload_image, tensor, session_token)
                print(f"[Seedance2] Image {idx} → {fname}")
                return fname
            except requests.exceptions.HTTPError as e:
                self._handle_upload_http_error(e, f"image {idx}")
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"SjinnSeedance2Node: network error uploading image {idx}: {e}") from e

        async def _upload_video_async(idx: int, video_obj) -> tuple:
            print(f"[Seedance2] Uploading video {idx}/{len(video_entries)}...")
            try:
                fname, dur = await loop.run_in_executor(
                    _POOL, self._upload_video_from_obj, video_obj, session_token)
                print(f"[Seedance2] Video {idx} → {fname}  ({dur:.3f}s)")
                return fname, dur
            except requests.exceptions.HTTPError as e:
                self._handle_upload_http_error(e, f"video {idx}")
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"SjinnSeedance2Node: network error uploading video {idx}: {e}") from e

        async def _upload_audio_async(idx: int, audio_obj) -> tuple:
            print(f"[Seedance2] Uploading audio {idx}/{len(audio_entries)}...")
            try:
                fname, dur = await loop.run_in_executor(
                    _POOL, self._upload_audio, audio_obj, session_token)
                print(f"[Seedance2] Audio {idx} → {fname}  ({dur:.3f}s)")
                return fname, dur
            except requests.exceptions.HTTPError as e:
                self._handle_upload_http_error(e, f"audio {idx}")
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"SjinnSeedance2Node: network error uploading audio {idx}: {e}") from e

        # Fire all uploads concurrently
        upload_tasks = (
            [_upload_image_async(i, t) for i, t in enumerate(images, 1)] +
            [_upload_video_async(i, v) for i, v in enumerate(video_entries, 1)] +
            [_upload_audio_async(i, a) for i, a in enumerate(audio_entries, 1)]
        )
        all_results = await asyncio.gather(*upload_tasks)
        pbar.set(38)

        # Split results back
        n_img = len(images)
        n_vid = len(video_entries)
        image_uuids    = list(all_results[:n_img])
        video_results  = list(all_results[n_img:n_img + n_vid])
        audio_results  = list(all_results[n_img + n_vid:])

        video_uuids     = [r[0] for r in video_results]
        video_durations = [r[1] for r in video_results]
        audio_uuids     = [r[0] for r in audio_results]
        audio_durations = [r[1] for r in audio_results]

        upload_elapsed = int(time.time() - generate_start)
        print(f"[Seedance2] All uploads done in {upload_elapsed}s — creating task...")

        # ── Create task ────────────────────────────────────────────────
        try:
            task_id = await loop.run_in_executor(
                _POOL,
                self._create_task,
                session_token, prompt, ratio, duration,
                image_uuids, video_uuids, video_durations,
                audio_uuids, audio_durations,
                speed_mode, accelerate_real_face, smart_rescue, seed,
            )
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 401:
                print("[Seedance2] AUTH ERROR — session token rejected (401).")
                print("[Seedance2] Refresh: sjinn.ai → F12 → Application → Cookies → __Secure-next-auth.session-token")
                raise RuntimeError("SjinnSeedance2Node: session token rejected (401) during task creation") from e
            raise RuntimeError(f"SjinnSeedance2Node: task creation HTTP error: {e}") from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"SjinnSeedance2Node: network error during task creation: {e}") from e

        print(f"[Seedance2] Task created — ID: {task_id}")
        pbar.set(42)

        # Deadline starts NOW (after uploads) — gives full budget for generation
        task_start = time.time()

        # ── Poll until done (async — yields during sleep, enables parallelism) ──
        video_url = await self._poll_async(task_id, api_key, duration, task_start, pbar, loop)
        pbar.set(90)

        # ── Download video ─────────────────────────────────────────────
        total_elapsed = int(time.time() - generate_start)
        print(f"[Seedance2] Downloading result — total elapsed {total_elapsed}s")
        video_path = await loop.run_in_executor(_POOL, self._download_video, video_url)
        pbar.set(95)

        # ── Extract frames ─────────────────────────────────────────────
        first_frame, last_frame, frames = await asyncio.gather(
            loop.run_in_executor(_POOL, self._get_frame_tensor, video_path, "first"),
            loop.run_in_executor(_POOL, self._get_frame_tensor, video_path, "last"),
            loop.run_in_executor(_POOL, self._load_all_frames, video_path),
        )
        pbar.set(100)

        if not _HAS_VIDEO_FROM_FILE:
            raise RuntimeError(
                "SjinnSeedance2Node: VideoFromFile not available — update ComfyUI to v0.14+")

        print(f"[Seedance2] Done — {video_path}")
        return (_VideoFromFile(video_path), first_frame, last_frame, frames)

    # ------------------------------------------------------------------
    # Async polling — await asyncio.sleep() so other async nodes run
    # during the wait intervals (enables parallel generation)
    # ------------------------------------------------------------------

    async def _poll_async(
        self,
        task_id: str,
        api_key: str,
        duration: str,
        task_start: float,
        pbar: "_PBar",
        loop,
    ) -> str:
        """Poll Sjinn API until task completes. Returns output video URL.
        Uses asyncio.sleep so other concurrent async nodes make progress
        while this one is waiting."""

        estimated_sec = int(duration) * 50
        deadline      = task_start + self.MAX_POLL_MINUTES * 60
        attempt       = 0

        print(
            f"[Seedance2] Polling — task_id={task_id} | "
            f"est. ~{estimated_sec}s | interval={self.POLL_INTERVAL_SEC}s | "
            f"timeout={self.MAX_POLL_MINUTES}min"
        )

        while time.time() < deadline:
            attempt += 1
            elapsed = time.time() - task_start
            pbar.set(42 + int(47 * min(elapsed / max(estimated_sec, 1), 1.0)))

            # Poll with retry on transient network errors
            data = None
            for retry in range(self.POLL_RETRY_LIMIT):
                try:
                    data = await loop.run_in_executor(
                        _POOL, self._check_status, task_id, api_key)
                    break
                except requests.exceptions.RequestException as e:
                    print(f"[Seedance2] Poll #{attempt} retry {retry+1}/{self.POLL_RETRY_LIMIT} — {e}")
                    if retry < self.POLL_RETRY_LIMIT - 1:
                        await asyncio.sleep(self.POLL_RETRY_DELAY)
                    else:
                        raise RuntimeError(
                            f"SjinnSeedance2Node: polling failed after {self.POLL_RETRY_LIMIT} "
                            f"retries (task {task_id}): {e}") from e

            status_data = data.get("data", {})
            status      = status_data.get("status", -99)
            print(f"[Seedance2] Poll #{attempt:03d}  elapsed={int(elapsed)}s  status={status}")

            if status == 1:
                output_urls = (
                    status_data.get("output_urls")
                    or status_data.get("output_url_list")
                    or ([status_data["output_url"]] if status_data.get("output_url") else None)
                )
                if not output_urls:
                    raise RuntimeError(
                        f"SjinnSeedance2Node: task complete but no output URL: {status_data}")
                video_url = output_urls[0]
                print(f"[Seedance2] Generation done in {int(elapsed)}s — {video_url}")
                return video_url

            elif status == 2:
                error_msg = status_data.get("error", "unknown error")
                raise RuntimeError(f"SjinnSeedance2Node: task failed: {error_msg}")

            # Yield to other asyncio tasks during sleep
            await asyncio.sleep(self.POLL_INTERVAL_SEC)

        raise RuntimeError(
            f"SjinnSeedance2Node: timed out after {self.MAX_POLL_MINUTES} minutes "
            f"(task {task_id}). Generation may still be running on Sjinn — "
            f"check sjinn.ai to retrieve the result manually.")

    # ------------------------------------------------------------------
    # Status check (blocking — runs in thread pool)
    # ------------------------------------------------------------------

    def _check_status(self, task_id: str, api_key: str) -> dict:
        resp = requests.post(
            self.STATUS_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"task_id": task_id},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Upload helpers
    # ------------------------------------------------------------------

    def _handle_upload_http_error(self, e: requests.exceptions.HTTPError, kind: str):
        if e.response is not None and e.response.status_code == 401:
            raise RuntimeError(
                f"SjinnSeedance2Node: session token rejected (401) during {kind} upload") from e
        raise RuntimeError(f"SjinnSeedance2Node: HTTP error during {kind} upload: {e}") from e

    def _get_signed_url(self, session_token: str, content_type: str) -> tuple:
        """Request a presigned R2 upload URL. Returns (file_name, signed_url)."""
        resp = requests.post(
            self.UPLOAD_URL,
            headers={
                "Cookie": f"__Secure-next-auth.session-token={session_token}",
                "Content-Type": "application/json",
            },
            json={"bucket_name": "comfy-online", "content_type": content_type},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(f"SjinnSeedance2Node: upload_file API error: {data.get('errorMsg')}")
        return data["data"]["file_name"], data["data"]["signed_url"]

    @staticmethod
    def _r2_put(signed_url: str, data: bytes, content_type: str, timeout: int = 300):
        """PUT bytes to R2 presigned URL."""
        resp = requests.put(
            signed_url, data=data,
            headers={"Content-Type": content_type}, timeout=timeout)
        if not resp.ok:
            print(f"[Seedance2] R2 PUT failed: {resp.status_code} — {resp.text[:200]}")
        resp.raise_for_status()

    def _upload_image(self, tensor, session_token: str) -> str:
        jpeg_bytes = self._tensor_to_jpeg_bytes(tensor)
        file_name, signed_url = self._get_signed_url(session_token, "image/jpeg")
        self._r2_put(signed_url, jpeg_bytes, "image/jpeg", timeout=60)
        return file_name

    def _upload_video_from_obj(self, video_obj, session_token: str) -> tuple:
        source = video_obj.get_stream_source()
        duration_seconds = float(video_obj.get_duration())
        print(f"[Seedance2] Video source duration: {duration_seconds:.3f}s")
        if isinstance(source, str):
            with open(source, "rb") as f:
                video_bytes = f.read()
        else:
            source.seek(0)
            video_bytes = source.read()
        file_name, signed_url = self._get_signed_url(session_token, "video/mp4")
        print(f"[Seedance2] Video upload — file: {file_name}")
        self._r2_put(signed_url, video_bytes, "video/mp4", timeout=300)
        return file_name, duration_seconds

    def _upload_audio(self, audio_dict: dict, session_token: str) -> tuple:
        waveform    = audio_dict["waveform"][0].cpu()    # [channels, samples]
        sample_rate = int(audio_dict["sample_rate"])
        n_channels  = waveform.shape[0]
        n_samples   = waveform.shape[1]
        dur         = float(n_samples) / sample_rate
        print(f"[Seedance2] Audio: {n_channels}ch, {sample_rate}Hz, {dur:.3f}s")
        pcm = (waveform.numpy().T * 32767).clip(-32768, 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        audio_bytes = buf.getvalue()
        file_name, signed_url = self._get_signed_url(session_token, "audio/wav")
        self._r2_put(signed_url, audio_bytes, "audio/wav", timeout=120)
        return file_name, dur

    # ------------------------------------------------------------------
    # Task creation
    # ------------------------------------------------------------------

    def _create_task(
        self,
        session_token: str,
        prompt: str,
        ratio: str,
        duration: str,
        image_uuids: list,
        video_uuids: list,
        video_durations: list,
        audio_uuids: list,
        audio_durations: list,
        speed_mode: str = "pro",
        accelerate_real_face: bool = True,
        smart_rescue: bool = False,
        seed: int = -1,
    ) -> str:
        task_input = {
            "prompt": prompt,
            "ratio": ratio,
            "duration": str(duration),
            "image_urls": image_uuids,
            "video_urls": video_uuids,
            "view_video_durations_1": video_durations,
            "audio_urls": audio_uuids,
            "view_audio_durations_2": audio_durations,
            "mode": speed_mode,
            "accelerate_real_face": accelerate_real_face,
            "smart_rescue": smart_rescue,
        }
        if seed >= 0:
            task_input["seed"] = seed
            print(f"[Seedance2] Fixed seed: {seed}")
        else:
            print("[Seedance2] Seed: random")

        payload = {"id": "seedance20-video", "input": task_input, "mode": "template"}
        print(f"[Seedance2] Task payload (prompt truncated): "
              f"images={len(image_uuids)}, videos={len(video_uuids)}, "
              f"audio={len(audio_uuids)}, ratio={ratio}, dur={duration}")

        resp = requests.post(
            self.CREATE_URL,
            headers={
                "Cookie": f"__Secure-next-auth.session-token={session_token}",
                "Content-Type": "application/json",
                "sjinn-version": "61",
            },
            json=payload,
            timeout=90,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(f"SjinnSeedance2Node: task creation failed: {data.get('errorMsg')}")
        return data["data"]["task_id"]

    # ------------------------------------------------------------------
    # Video / frame helpers
    # ------------------------------------------------------------------

    def _download_video(self, video_url: str) -> str:
        save_dir = os.path.join(folder_paths.get_output_directory(), "api_s2", "raw")
        os.makedirs(save_dir, exist_ok=True)
        counter = 1
        while True:
            path = os.path.join(save_dir, f"raw_video_{counter:05d}.mp4")
            if not os.path.exists(path):
                break
            counter += 1
        resp = requests.get(video_url, stream=True, timeout=300)
        resp.raise_for_status()
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
        size_kb = os.path.getsize(path) // 1024
        print(f"[Seedance2] Saved — {path} ({size_kb} KB)")
        return path

    def _tensor_to_jpeg_bytes(self, tensor) -> bytes:
        arr = tensor.squeeze(0).cpu().numpy()
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        pil = PilImage.fromarray(arr).convert("RGB")
        w, h = pil.size
        print(f"[Seedance2] Image size: {w}x{h}")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def _get_frame_tensor(self, video_path: str, position: str = "first"):
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if position == "last":
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(total - 1, 0))
            ret, frame = cap.read()
            cap.release()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                arr = rgb.astype(np.float32) / 255.0
                print(f"[Seedance2] {position.capitalize()} frame: {arr.shape[1]}x{arr.shape[0]}px")
                return torch.from_numpy(arr).unsqueeze(0)
            print(f"[Seedance2] WARNING — could not read {position} frame")
        except ImportError:
            print("[Seedance2] WARNING — opencv-python not installed; frame outputs are blank")
        except Exception as e:
            print(f"[Seedance2] WARNING — frame extraction failed: {e}")
        return torch.zeros(1, 64, 64, 3)

    def _load_all_frames(self, video_path: str) -> torch.Tensor:
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            cap.release()
            if frames:
                print(f"[Seedance2] Loaded {len(frames)} frames as IMAGE batch")
                return torch.from_numpy(np.stack(frames, axis=0))
            print("[Seedance2] WARNING — no frames loaded")
        except ImportError:
            print("[Seedance2] WARNING — opencv-python not installed; frames output is blank")
        except Exception as e:
            print(f"[Seedance2] WARNING — frames load failed: {e}")
        return torch.zeros(1, 64, 64, 3)


NODE_CLASS_MAPPINGS       = {"SjinnSeedance2Node": SjinnSeedance2Node}
NODE_DISPLAY_NAME_MAPPINGS = {"SjinnSeedance2Node": "🎬 Seedance 2.0 (Sjinn.ai)"}
