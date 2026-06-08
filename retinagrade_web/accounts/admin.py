from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser


@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    list_display = ('username', 'email', 'role', 'institution', 'is_active', 'date_joined')
    list_filter = ('role', 'is_active', 'is_staff')
    search_fields = ('username', 'email', 'first_name', 'last_name', 'institution')
    fieldsets = UserAdmin.fieldsets + (
        ('RetinaGrade', {'fields': ('role', 'institution')}),
    )
    add_fieldsets = UserAdmin.add_fieldsets + (
        ('RetinaGrade', {'fields': ('role', 'institution')}),
    )
