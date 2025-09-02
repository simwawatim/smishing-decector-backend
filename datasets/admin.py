from django.contrib import admin
from .models import SMSMessage

@admin.register(SMSMessage)
class SMSMessageAdmin(admin.ModelAdmin):
    list_display = ('id', 'label', 'message', 'created_at')
    search_fields = ('message',)
    list_filter = ('label',)