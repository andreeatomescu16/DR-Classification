from django.conf import settings
from django.db import models
from django.utils import timezone


GRADE_LABELS = {
    0: 'No DR',
    1: 'Mild NPDR',
    2: 'Moderate NPDR',
    3: 'Severe NPDR',
    4: 'Proliferative DR',
}

GRADE_DESCRIPTIONS = {
    0: (
        'No lesions detected. The fundus appears within normal limits. '
        'Routine annual screening is recommended to monitor for any early '
        'development of retinal changes.'
    ),
    1: (
        'Microaneurysms present with possible small retinal haemorrhages. '
        'Early diabetic changes detected in the retinal vasculature. '
        'Close monitoring and glycaemic control are recommended.'
    ),
    2: (
        'Retinal haemorrhages, microaneurysms, and hard exudates present. '
        'Clinically significant macular oedema may be developing. '
        'Ophthalmology referral within 3 months is advised.'
    ),
    3: (
        'Multiple retinal haemorrhages, venous beading, and intraretinal '
        'microvascular abnormalities (IRMA) identified. High risk of '
        'progression to proliferative disease. Urgent referral required.'
    ),
    4: (
        'Neovascularisation of the disc (NVD) or elsewhere (NVE) detected. '
        'Risk of vitreous haemorrhage is elevated. Immediate ophthalmological '
        'intervention — laser or anti-VEGF — is required.'
    ),
}


class Prediction(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='predictions',
    )
    image = models.ImageField(upload_to='fundus/%Y/%m/%d/')
    grade = models.IntegerField()
    label = models.CharField(max_length=50)
    confidence = models.FloatField()
    probabilities = models.JSONField()
    grad_cam_image = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'Prediction #{self.pk} — {self.label} ({self.user.username})'

    def grade_label(self):
        return GRADE_LABELS.get(self.grade, 'Unknown')

    def grade_description(self):
        return GRADE_DESCRIPTIONS.get(self.grade, '')

    def confidence_pct(self):
        return round(self.confidence * 100, 1)

    def needs_referral(self):
        return self.grade >= 2

    def safe_grad_cam(self):
        """Strip any characters outside the base64 alphabet before rendering."""
        import re
        return re.sub(r'[^A-Za-z0-9+/=]', '', self.grad_cam_image)


class AnalysisSession(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='sessions',
    )
    title = models.CharField(max_length=200, default='New Session')
    predictions = models.ManyToManyField(Prediction, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'Session "{self.title}" — {self.user.username}'


class RateLimit(models.Model):
    """Simple DB-based rate limiter for the /analyse/ endpoint."""
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='rate_limit',
    )
    count = models.IntegerField(default=0)
    window_start = models.DateTimeField(default=timezone.now)

    def reset_if_expired(self):
        if (timezone.now() - self.window_start).total_seconds() >= 3600:
            self.count = 0
            self.window_start = timezone.now()
            self.save(update_fields=['count', 'window_start'])

    def is_allowed(self, limit: int) -> bool:
        self.reset_if_expired()
        return self.count < limit

    def increment(self):
        self.count += 1
        self.save(update_fields=['count'])
