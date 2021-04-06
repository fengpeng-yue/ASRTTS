import argparse
parser = argparse.ArgumentParser()
parser.add_argument("text", type=str, help="text to be uppered")
parser.add_argument("output_text", type=str, help="uppered text")
args = parser.parse_args()
output_f = open(args.output_text,"w")
with open(args.text,'r') as f:
    items = f.read().splitlines()
    for item in items:
        if len(item.split(" ",1)) < 2:
            print(item)
            continue
        utt,content = item.split(" ",1)
        content = content.upper()
        out_item = utt + " " + content + "\n"
        output_f.write(out_item)
    f.close()
output_f.close()
