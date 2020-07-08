sudo apt install git -y
sudo apt install screen -y
sudo apt install python3-pip -y
sudo apt install python3-venv -y
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
mv ./AWS_scripts/core.py venv/lib/python3.6/site-packages/xgboost/core.py