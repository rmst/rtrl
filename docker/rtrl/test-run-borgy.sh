echo "THIS IS OUTDATED! Exiting..."
exit

tmp=$(mktemp -d)
mkdir $tmp/e
mkdir $tmp/l

PATH="$(pwd)/bin:$PATH"
EXPERIMENTS=$tmp/e LOGS=$tmp/l BORGY_JOB_ID=3 run-borgy 2 3 '"teststr"' ' '

echo "exit code is $?"

# mv $tmp test
rm -r $tmp