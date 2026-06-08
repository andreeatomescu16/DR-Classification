from django.contrib.auth.models import AbstractUser
from django.db import models


class CustomUser(AbstractUser):
    ROLE_PATIENT = 'patient'
    ROLE_CLINICIAN = 'clinician'
    ROLE_ADMIN = 'admin'

    ROLE_CHOICES = [
        (ROLE_PATIENT, 'Patient'),
        (ROLE_CLINICIAN, 'Clinician'),
        (ROLE_ADMIN, 'Admin'),
    ]

    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default=ROLE_PATIENT)
    institution = models.CharField(max_length=200, blank=True)

    def is_clinician(self):
        return self.role == self.ROLE_CLINICIAN

    def is_admin_role(self):
        return self.role == self.ROLE_ADMIN

    def get_initials(self):
        parts = (self.get_full_name() or self.username).split()
        if len(parts) >= 2:
            return (parts[0][0] + parts[-1][0]).upper()
        return self.username[:2].upper()

    def __str__(self):
        return f'{self.username} ({self.get_role_display()})'
