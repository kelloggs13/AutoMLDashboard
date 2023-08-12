import requests

# Replace these with your own values
client_id = '3a477dfd-ee3c-4044-8159-97b0b98281c8'
client_secret = '2e8af7e9-82df-44d6-85ec-42762454b50c'
redirect_uri = '127.0.0.1'
authorization_code = 'authorization_code'

token_url = 'https://login.microsoftonline.com/common/oauth2/token'

token_data = {
    'grant_type': 'authorization_code',
    'client_id': client_id,
    'client_secret': client_secret,
    'code': authorization_code,
    'redirect_uri': redirect_uri,
}

# Get access token
response = requests.post(token_url, data=token_data)
access_token = response.json().get('access_token')

# Make API request to get Drive ID
drive_url = 'https://graph.microsoft.com/v1.0/me/drive'
headers = {
    'Authorization': f'Bearer {access_token}'
}

drive_info = requests.get(drive_url, headers=headers).json()
drive_id = drive_info.get('id')

print(f"Drive ID: {drive_id}")
