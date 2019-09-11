find_master_rev(){
  tempdir=$(mktemp -d)
  git clone $1 $tempdir/repo -q
  git -C $tempdir/repo rev-parse origin/master
  rm -rf $tempdir
}

set -e  # exit on error

AVENUE_REV=${1:-$(find_master_rev git@github.com:ElementAI/Avenue.git)}
echo "Building with Avenue revision $AVENUE_REV"

git clone git@github.com:ElementAI/Avenue.git avenue
pushd avenue
  git reset --hard ${AVENUE_REV}
popd

docker build -t unity3d - < unity3d.dockerfile  # TODO: this image should be temporary
docker build -t unity3d-py --build-arg BASE=unity3d:latest --build-arg AVENUE_REV=$AVENUE_REV ../pytorch
docker build -t $DOCKER_REMOTE/avenue:$AVENUE_REV --build-arg BASE=unity3d-py:latest . 
docker push $DOCKER_REMOTE/avenue:$AVENUE_REV

rm -rf avenue