#!/bin/sh
# Rebuild and publish the dashboard on the server. Run from ~/proofbench there,
# after syncing the tree.
#
# The export stage runs the test suite, the replay gate and the dashboard build
# inside the image, so a failing test or a grader that moved means nothing is
# published - the old page stays up.
set -eu

docker build --target runtime -t proofbench:latest .
rm -rf out
docker build --target export --output type=local,dest=out .

mkdir -p ~/edge-sites/proofbench
cp out/index.html ~/edge-sites/proofbench/index.html.new
chmod 644 ~/edge-sites/proofbench/index.html.new
# Swapped in one rename, so a reader never gets half a file.
mv ~/edge-sites/proofbench/index.html.new ~/edge-sites/proofbench/index.html

# The site file is installed once (see deploy/proofbench.caddy); reload only if
# it changed.
if ! cmp -s deploy/proofbench.caddy ~/edge-sites/proofbench.caddy; then
  cp deploy/proofbench.caddy ~/edge-sites/proofbench.caddy
  docker exec clearhire-caddy-1 caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
  docker exec clearhire-caddy-1 caddy reload --config /etc/caddy/Caddyfile --adapter caddyfile
fi

docker builder prune -f --filter until=1h >/dev/null
echo "published $(wc -c < out/index.html) bytes"
