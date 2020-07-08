from Utils.TelegramBot import telegram_bot_send_update

import requests

import time

r = requests.get("https://recsys-twitter.com/leaderboard/latest")

result = str(r.content)

groupname = "luchetto"
start_index = result.find(groupname)

print(start_index)

toparse = result[start_index:start_index+350]

print(toparse)

splitted = toparse.split("<td>")

for i in range(0,10):
    splitted[i] = splitted[i][0:splitted[i].index('<')]
    print(splitted[i])

praucs= [splitted[2], splitted[4], splitted[6], splitted[8]]
rces = [splitted[3], splitted[5], splitted[7], splitted[9]]

classes = ["retweet","reply", "like", "rt with comment"]

for c in classes:
    i = 0
    print(f"{c}: PRAUC:\t{praucs[i]}, RCE:\t{rces[i]}")
    i+=1

while True:
    r = requests.get("https://recsys-twitter.com/leaderboard/latest")

    result = str(r.content)

    start_index = result.find(groupname)

    print(start_index)

    toparse = result[start_index:start_index+350]

    print(toparse)

    splitted = toparse.split("<td>")

    for i in range(0,10):
        splitted[i] = splitted[i][0:splitted[i].index('<')]
        print(splitted[i])

    new_praucs= [splitted[2], splitted[4], splitted[6], splitted[8]]
    new_rces = [splitted[3], splitted[5], splitted[7], splitted[9]]

    for i in range(0,4):
        if new_praucs[i] != praucs[i]:
            telegram_bot_send_update(f"{groupname}\n{classes[i]}:\nPREVIOUS RESULTS: \nPRAUC:\t{praucs[i]}, RCE:\t{rces[i]}\nNEW RESULTS: \nPRAUC:\t{new_praucs[i]}, RCE:\t{new_rces[i]}")
            praucs[i] = new_praucs[i]
            rces[i] = new_rces[i]

    time.sleep(10)
