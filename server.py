import argparse
import json

import spu.utils.distributed as ppd
from spu.utils.polyfill import Process

parser = argparse.ArgumentParser(description="SPU node service.")
parser.add_argument(
    "-c", "--config", default="conf/3pc.json", help="Config file"
)
subparsers = parser.add_subparsers(dest='command')
parser_start = subparsers.add_parser('start', help='to start a single node')
parser_start.add_argument("-n", "--node_id", default="node:0", help="the node id")
parser_up = subparsers.add_parser('up', help='to bring up all nodes')
parser_list = subparsers.add_parser('list', help='list node information')
parser_list.add_argument("-v", "--verbose", action='store_true', help="verbosely")

if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        conf = json.load(file)

    nodes_def = conf["nodes"]
    devices_def = conf["devices"]

    if args.command == "start":
        ppd.RPC.serve(args.node_id, nodes_def)
    elif args.command == "up":
        workers = []
        for node_id in nodes_def.keys():
            worker = Process(target=ppd.RPC.serve, args=(node_id, nodes_def))
            worker.start()
            workers.append(worker)

        for worker in workers:
            worker.join()
    else:
        parser.print_help()
