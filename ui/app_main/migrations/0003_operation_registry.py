from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('app_main', '0002_english_ui'),
    ]

    operations = [
        migrations.AlterField(
            model_name='job',
            name='operation',
            field=models.CharField(max_length=24),
        ),
    ]
