from app import app

c = app.test_client()
resp = c.post('/predict', data={
    'T':'7.4', 'TM':'9.8', 'Tm':'4.8', 'SLP':'1017.6', 'H':'93.0', 'VV':'0.5', 'V':'4.3', 'VM':'9.4'
})
print('status_code=', resp.status_code)
print(resp.data.decode())
