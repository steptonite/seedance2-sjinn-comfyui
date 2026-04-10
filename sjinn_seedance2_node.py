import base64
import binascii
import io
import os
import struct
import time
import wave
import requests
import numpy as np
import torch
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


class SjinnSeedance2Node:
    """ComfyUI node: Sjinn.ai Seedance 2.0 video generation.
    Accepts up to 6 reference images, up to 3 reference videos, up to 2 reference audio clips.
    Downloads result and returns native VIDEO + first_frame + last_frame."""

    RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4"]
    DURATIONS = ["4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
    SPEED_MODES = ["pro", "fast"]

    BASE_URL = "https://sjinn.ai"
    UPLOAD_URL = f"{BASE_URL}/api/upload_file"
    CREATE_URL = f"{BASE_URL}/api/create_seedance20_video"
    STATUS_URL = f"{BASE_URL}/api/un-api/query_tool_task_status"

    POLL_INTERVAL_SECONDS = 15
    MAX_POLL_MINUTES = 15

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
                # Reference images — mention @Image1 @Image2 ... in prompt
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                # Reference videos — connect Load Video node, mention @Video1 @Video2 ... in prompt
                # NOTE: Seedance requires reference video ≤ 720P and ≥ 2 seconds
                "video_1": ("VIDEO", {"tooltip": "Connect Load Video node. Use @Video1 in prompt."}),
                "video_2": ("VIDEO", {"tooltip": "Connect Load Video node. Use @Video2 in prompt."}),
                "video_3": ("VIDEO", {"tooltip": "Connect Load Video node. Use @Video3 in prompt."}),
                # Reference audio clips — connect Load Audio node (sjinn.ai supports max 2)
                "audio_1": ("AUDIO", {"tooltip": "Reference audio clip. Connect Load Audio node."}),
                "audio_2": ("AUDIO", {"tooltip": "Reference audio clip. Connect Load Audio node."}),
                # Generation settings
                "ratio":           (cls.RATIOS,      {"default": "16:9"}),
                "duration":        (cls.DURATIONS,   {"default": "5"}),
                "speed_mode":      (cls.SPEED_MODES, {"default": "pro", "tooltip": "pro = higher quality / slower, fast = lower quality / quicker"}),
                "accelerate_real_face": ("BOOLEAN", {"default": True,  "tooltip": "Preserve face consistency across frames (accelerate_real_face)"}),
                "smart_rescue":    ("BOOLEAN", {"default": True,  "tooltip": "Auto-rewrite prompt if it violates content policy. Disable to use prompt verbatim."}),
                "seed":            ("INT",     {"default": -1, "min": -1, "max": 2**32 - 1, "tooltip": "-1 = random seed, 0–4294967295 = fixed seed"}),
            },
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("video", "first_frame", "last_frame", "frames")
    FUNCTION = "generate"
    CATEGORY = "Sjinn.ai"

    # ------------------------------------------------------------------
    # Progress bar helper
    # ------------------------------------------------------------------
    class _PBar:
        def __init__(self, total: int):
            self._cb = _ComfyProgressBar(total) if _HAS_PROGRESS_BAR else None
            self._total = total
            self._cur = 0

        def set(self, value: int):
            value = max(self._cur, min(value, self._total))
            self._cur = value
            if self._cb is not None:
                self._cb.update_absolute(value, self._total)

        def advance(self, delta: int = 1):
            self.set(self._cur + delta)

    # ------------------------------------------------------------------

    def generate(
        self,
        session_token: str,
        api_key: str,
        prompt: str,
        image_1=None, image_2=None, image_3=None,
        image_4=None, image_5=None, image_6=None,
        video_1=None, video_2=None, video_3=None,
        audio_1=None, audio_2=None,
        ratio: str = "16:9",
        duration: str = "5",
        speed_mode: str = "pro",
        accelerate_real_face: bool = True,
        smart_rescue: bool = True,
        seed: int = -1,
    ):
        if not session_token.strip():
            raise ValueError("SjinnSeedance2Node: session_token is required")
        if not api_key.strip():
            raise ValueError("SjinnSeedance2Node: api_key is required")

        images        = [x for x in [image_1, image_2, image_3, image_4, image_5, image_6] if x is not None]
        video_entries = [x for x in [video_1, video_2, video_3] if x is not None]
        audio_entries = [x for x in [audio_1, audio_2] if x is not None]

        img_count = len(images)
        vid_count = len(video_entries)
        aud_count = len(audio_entries)

        print(
            f"[Seedance2] Starting — {img_count} images, {vid_count} videos, {aud_count} audio | "
            f"ratio={ratio}, duration={duration}s, speed_mode={speed_mode}, "
            f"accelerate_real_face={accelerate_real_face}, smart_rescue={smart_rescue}, seed={seed}"
        )
        start_time = time.time()

        # Progress bar layout (0-100):
        #   0 – 20  : upload images
        #  20 – 32  : upload videos
        #  32 – 38  : upload audio
        #  38 – 42  : create task
        #  42 – 90  : generation polling (time-based estimate)
        #  90 – 95  : download video
        #  95 – 100 : extract frames
        pbar = self._PBar(100)

        # ── Upload images ──────────────────────────────────────────────
        image_uuids = []
        for idx, tensor in enumerate(images, start=1):
            print(f"[Seedance2] Uploading image {idx}/{img_count}...")
            try:
                fname = self._upload_image(tensor, session_token)
                image_uuids.append(fname)
                print(f"[Seedance2] Image {idx} → {fname}")
            except requests.exceptions.HTTPError as e:
                self._handle_upload_http_error(e, "image")
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"SjinnSeedance2Node: network error during image upload: {e}") from e
            pbar.set(round(20 * idx / max(img_count, 1)))

        # ── Upload videos ──────────────────────────────────────────────
        video_uuids = []
        video_durations = []
        for idx, video_obj in enumerate(video_entries, start=1):
            print(f"[Seedance2] Uploading video {idx}/{vid_count}...")
            try:
                fname, dur = self._upload_video_from_obj(video_obj, session_token)
                video_uuids.append(fname)
                video_durations.append(dur)
                print(f"[Seedance2] Video {idx} → {fname}  ({dur:.3f}s)")
            except requests.exceptions.HTTPError as e:
                self._handle_upload_http_error(e, "video")
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"SjinnSeedance2Node: network error during video upload: {e}") from e
            pbar.set(20 + round(12 * idx / max(vid_count, 1)))

        # ── Upload audio ───────────────────────────────────────────────
        audio_uuids = []
        audio_durations = []
        for idx, audio_obj in enumerate(audio_entries, start=1):
            print(f"[Seedance2] Uploading audio {idx}/{aud_count}...")
            try:
                fname, dur = self._upload_audio(audio_obj, session_token)
                audio_uuids.append(fname)
                audio_durations.append(dur)
                print(f"[Seedance2] Audio {idx} → {fname}  ({dur:.3f}s)")
            except requests.exceptions.HTTPError as e:
                self._handle_upload_http_error(e, "audio")
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"SjinnSeedance2Node: network error during audio upload: {e}") from e
            pbar.set(32 + round(6 * idx / max(aud_count, 1)))

        pbar.set(38)

        # ── Create task ────────────────────────────────────────────────
        print("[Seedance2] Creating task...")
        try:
            task_id = self._create_task(
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

        # ── Poll until done ────────────────────────────────────────────
        video_url = self._poll(task_id, api_key, duration, start_time, pbar)
        pbar.set(90)

        # ── Download video ─────────────────────────────────────────────
        elapsed_total = int(time.time() - start_time)
        print(f"[Seedance2] Downloading result video — total elapsed {elapsed_total}s")
        video_path = self._download_video(video_url)
        pbar.set(95)

        # ── Extract frames ─────────────────────────────────────────────
        first_frame = self._get_frame_tensor(video_path, position="first")
        last_frame  = self._get_frame_tensor(video_path, position="last")
        frames      = self._load_all_frames(video_path)
        pbar.set(100)

        # ── Build VIDEO output ─────────────────────────────────────────
        if _HAS_VIDEO_FROM_FILE:
            video_out = _VideoFromFile(video_path)
        else:
            raise RuntimeError(
                "SjinnSeedance2Node: VideoFromFile not available — update ComfyUI to v0.14+"
            )

        print(f"[Seedance2] Done — {video_path}")
        return (video_out, first_frame, last_frame, frames)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_upload_http_error(self, e: requests.exceptions.HTTPError, kind: str):
        if e.response is not None and e.response.status_code == 401:
            raise RuntimeError(f"SjinnSeedance2Node: session token rejected (401) during {kind} upload") from e
        raise RuntimeError(f"SjinnSeedance2Node: HTTP error during {kind} upload: {e}") from e

    def _download_video(self, video_url: str) -> str:
        """Download output video to ComfyUI output folder. Returns local file path."""
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

    def _load_all_frames(self, video_path: str) -> torch.Tensor:
        """Load all video frames as IMAGE tensor [B,H,W,3] float32 0-1. For VHS/frame processing."""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb.astype(np.float32) / 255.0)
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

    def _get_frame_tensor(self, video_path: str, position: str = "first"):
        """Extract first or last frame from video file. Returns IMAGE tensor [1,H,W,3] float32 0-1."""
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

    def _tensor_to_jpeg_bytes(self, tensor) -> bytes:
        arr = tensor.squeeze(0).cpu().numpy()
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        pil = PilImage.fromarray(arr).convert("RGB")
        w, h = pil.size
        print(f"[Seedance2] Image size: {w}x{h}")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    @staticmethod
    def _r2_put(signed_url: str, data: bytes, content_type: str, timeout: int = 300):
        """PUT bytes to R2 presigned URL with error logging."""
        resp = requests.put(signed_url, data=data, headers={"Content-Type": content_type}, timeout=timeout)
        if not resp.ok:
            print(f"[Seedance2] R2 PUT failed: {resp.status_code} — {resp.text[:200]}")
        resp.raise_for_status()

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
        bucket = signed_url.split("//")[1].split(".")[0]
        print(f"[Seedance2] Video upload — bucket: {bucket}, file: {file_name}")
        self._r2_put(signed_url, video_bytes, "video/mp4", timeout=300)
        return file_name, duration_seconds

    def _upload_audio(self, audio_dict: dict, session_token: str) -> tuple:
        """Upload AUDIO dict (ComfyUI native) → R2 as WAV. Returns (filename, duration_seconds)."""
        waveform = audio_dict["waveform"]   # [batch, channels, samples]
        sample_rate = int(audio_dict["sample_rate"])
        waveform = waveform[0].cpu()        # [channels, samples]
        n_channels = waveform.shape[0]
        n_samples  = waveform.shape[1]
        duration_seconds = float(n_samples) / sample_rate
        print(f"[Seedance2] Audio: {n_channels}ch, {sample_rate}Hz, {duration_seconds:.3f}s")

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
        return file_name, duration_seconds

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
        smart_rescue: bool = True,
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
        print(f"[Seedance2] Task payload: {payload}")

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

    def _poll(self, task_id: str, api_key: str, duration: str, start_time: float, pbar: "_PBar") -> str:
        """Poll until complete. Returns output video URL. Progress 42 → 90."""
        estimated_seconds = int(duration) * 50
        deadline = start_time + self.MAX_POLL_MINUTES * 60
        attempt = 0

        print(
            f"[Seedance2] Polling — task_id={task_id} | "
            f"output={duration}s | est. ~{estimated_seconds}s | interval={self.POLL_INTERVAL_SECONDS}s"
        )

        while time.time() < deadline:
            attempt += 1
            elapsed = time.time() - start_time

            frac = min(elapsed / max(estimated_seconds, 1), 1.0)
            pbar.set(42 + int(47 * frac))

            try:
                resp = requests.post(
                    self.STATUS_URL,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"task_id": task_id},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"SjinnSeedance2Node: network error during polling: {e}") from e

            status_data = data.get("data", {})
            status = status_data.get("status", -99)
            print(f"[Seedance2] Poll #{attempt:03d}  elapsed={int(elapsed)}s  status={status}")

            if status == 1:
                output_urls = (
                    status_data.get("output_urls")
                    or status_data.get("output_url_list")
                    or ([status_data["output_url"]] if status_data.get("output_url") else None)
                )
                if not output_urls:
                    raise RuntimeError(f"SjinnSeedance2Node: task complete but no output URL: {status_data}")
                video_url = output_urls[0]
                pbar.set(90)
                print(f"[Seedance2] Generation done — {int(elapsed)}s | {video_url}")
                return video_url

            elif status == 2:
                error_msg = status_data.get("error", "unknown error")
                raise RuntimeError(f"SjinnSeedance2Node: task failed: {error_msg}")

            else:
                time.sleep(self.POLL_INTERVAL_SECONDS)

        raise RuntimeError(
            f"SjinnSeedance2Node: timed out after {self.MAX_POLL_MINUTES} minutes (task {task_id})"
        )


NODE_CLASS_MAPPINGS = {"SjinnSeedance2Node": SjinnSeedance2Node}
NODE_DISPLAY_NAME_MAPPINGS = {"SjinnSeedance2Node": "🎬 Seedance 2.0 (Sjinn.ai)"}
