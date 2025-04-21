from python:3.13-alpine
copy requirements.txt /tmp
run pip install -r /tmp/requirements.txt; rm -f /tmp/requirements.txt
run mkdir /app
add bot.py /app
entrypoint ["python3", "/app/bot.py"]
