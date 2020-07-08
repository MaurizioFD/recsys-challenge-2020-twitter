
# setup ssh-agent for github repo key

eval "$(ssh-agent -s)"
ssh-add /home/ubuntu/data/.ssh
