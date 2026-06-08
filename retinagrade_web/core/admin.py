from django.contrib import admin
from .models import Prediction, AnalysisSession, RateLimit


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'label', 'grade', 'confidence_pct', 'created_at')
    list_filter = ('grade', 'created_at')
    search_fields = ('user__username', 'label', 'notes')
    readonly_fields = ('created_at', 'grade', 'label', 'confidence', 'probabilities', 'grad_cam_image')
    date_hierarchy = 'created_at'

    def confidence_pct(self, obj):
        return f'{obj.confidence_pct()}%'
    confidence_pct.short_description = 'Confidence'


@admin.register(AnalysisSession)
class AnalysisSessionAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'title', 'created_at', 'prediction_count')
    search_fields = ('user__username', 'title')
    date_hierarchy = 'created_at'

    def prediction_count(self, obj):
        return obj.predictions.count()
    prediction_count.short_description = 'Predictions'


@admin.register(RateLimit)
class RateLimitAdmin(admin.ModelAdmin):
    list_display = ('user', 'count', 'window_start')
    search_fields = ('user__username',)
