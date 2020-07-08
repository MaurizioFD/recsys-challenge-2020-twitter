import RootPath


def create_submission_file(tweets, users, predictions, output_file):


    # Preliminary checks
    assert len(tweets) == len(users), f"length are different tweets -> {len(tweets)}, and users -> {len(users)} "
    assert len(users) == len(predictions), f"length are different users -> {len(users)}, and predictions -> {len(predictions)} "
    assert len(tweets) == len(predictions), f"length are different tweets -> {len(tweets)}, and predictions -> {len(predictions)} "

    file = open(RootPath.get_root().joinpath(output_file), "w")

    for i in range(len(tweets)):
        file.write(f"{tweets[i]},{users[i]},{round(predictions[i], 4)}\n")

    file.close()