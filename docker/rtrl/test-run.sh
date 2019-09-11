echo "THIS FILE IS OUTDATED. Exiting..."
exit

tmp=$(mktemp -d)

PATH="$(pwd)/bin:$PATH"
run $tmp/e 1 '"teststr"' ' '

echo "exit code is $?"

# mv $tmp test
rm -r $tmp