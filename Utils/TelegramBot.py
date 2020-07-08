#----------------------------------------------------------------
#                       TELEGRAM BOT
#----------------------------------------------------------------


import requests

def telegram_bot_send_update(update):
	
	bot_token='1188007388:AAFjZeeKPkxocN-AtnfICXbOZi_0cjTUpiU'
	group_id='-479667758'
	
	send_update='https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + group_id + '&sparse_mode=Markdown&text=' + update
	#send_update = 'https://api.telegram.org/bot' + bot_token +'/getUpdates'
	response = requests.get(send_update)
	
	return response.json()