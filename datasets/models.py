from django.db import models


class SMSMessage(models.Model):
    LABEL_CHOICES = [
        ('scam', 'Scam'),
        ('ham', 'Ham'),
    ]
    
    label = models.CharField(max_length=10, choices=LABEL_CHOICES)
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.label}: {self.message[:50]}"
