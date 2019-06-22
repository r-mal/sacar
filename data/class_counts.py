import json
import sys
from collections import Counter
from pprint import PrettyPrinter


def boundary():
  data = json.load(open(sys.argv[1]))
  counts = Counter()
  for jdoc in data:
    for sentence in jdoc['sentences']:
      counts.update(sentence['boundary_labels'])
  pp = PrettyPrinter(indent=2)
  pp.pprint(dict(counts.iteritems()))


def attribute():
  import os
  from collections import defaultdict

  data = json.load(open(sys.argv[1]))
  outdir = sys.argv[2]
  counts_dict = defaultdict(Counter)
  for jdoc in data:
    for sentence in jdoc['sentences']:
      for concept in sentence['concepts']:
        for attr_name, label in concept[1].iteritems():
          counts_dict[attr_name].update([label])

  pp = PrettyPrinter(indent=2)
  for attr_name, counts in counts_dict.items():
    total = float(sum([v for v in counts.values()]))
    num_classes = float(len(counts))
    weight_dict = {}
    for cl_name, count in counts.iteritems():
      weight_dict[cl_name] = total / (num_classes * count)
    json.dump(weight_dict, open(os.path.join(outdir, "%s.json" % attr_name), 'w+'))
    print(attr_name)
    pp.pprint(weight_dict)


if __name__ == '__main__':
  attribute()
