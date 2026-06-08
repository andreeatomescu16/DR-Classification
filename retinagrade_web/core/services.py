import re
import requests
from django.conf import settings


class ServiceUnavailableError(Exception):
    pass


class PredictionService:
    _BASE_URL = None

    @classmethod
    def _base(cls) -> str:
        if cls._BASE_URL is None:
            cls._BASE_URL = settings.FASTAPI_BASE_URL.rstrip('/')
        return cls._BASE_URL

    def predict(self, image_file) -> dict:
        url = f'{self._base()}/predict'
        try:
            resp = requests.post(
                url,
                files={'file': (image_file.name, image_file, image_file.content_type)},
                timeout=30,
            )
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            raise ServiceUnavailableError('The analysis service timed out. Please try again.')
        except requests.exceptions.ConnectionError:
            raise ServiceUnavailableError('Cannot reach the analysis service. Ensure the FastAPI backend is running.')
        except requests.exceptions.HTTPError as exc:
            raise ServiceUnavailableError(f'Analysis service returned an error: {exc.response.status_code}')

        data = resp.json()
        self._validate(data)
        # FastAPI returns key 'gradcam_image' (no underscore between grad/cam)
        raw_gcam = data.get('gradcam_image') or data.get('grad_cam_image') or ''
        data['grad_cam_image'] = self._sanitise_b64(raw_gcam)
        return data

    def health_check(self) -> bool:
        try:
            resp = requests.get(f'{self._base()}/health', timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    @staticmethod
    def _validate(data: dict):
        required = ('grade', 'label', 'probabilities')
        for key in required:
            if key not in data:
                raise ServiceUnavailableError(f'Malformed response from analysis service: missing "{key}".')
        if not (0 <= int(data['grade']) <= 4):
            raise ServiceUnavailableError('Grade value out of range.')
        if len(data['probabilities']) != 5:
            raise ServiceUnavailableError('Expected 5 probability values.')

    @staticmethod
    def _sanitise_b64(s: str) -> str:
        return re.sub(r'[^A-Za-z0-9+/=]', '', s)
