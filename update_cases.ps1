$body = @{
    since_days = 7
} | ConvertTo-Json

Invoke-WebRequest `
    -Uri "http://127.0.0.1:8000/update_cases" `
    -Method POST `
    -Body $body `
    -ContentType "application/json"
