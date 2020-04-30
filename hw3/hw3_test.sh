# hw3_test.sh
#!/bin/bash

# download testing photo
curl -O testing https://drive.google.com/open?id=1PUdFfsbOUYvnhNsfA4krkftF4FinIt_y

# $1 prediction file name
python hw3_test.py $1

