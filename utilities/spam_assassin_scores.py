"""Utility to query online spam assassin instances for email scores"""
import sys
import os
# import time
import json
import requests


def score_email(email):
    """Query an online spam checker"""
    with open(email, 'r') as f_in:
        msg = f_in.read()

    resp = requests.post(
        'http://spamcheck.postmarkapp.com/filter',
        data={'email': msg, 'options': 'short'})
    return json.loads(resp.text)['score']


def main(args):
    """Run over a bunch of directories"""
    scores = {}
    for arg in args:
        for root, _, files in os.walk(arg):
            for email in files:
                path = os.path.join(root, email)
                scores[path] = score_email(path)
                print " [+] Scored %s" % email
                # time.sleep(0.0001)

    with open('sa_scores.json', 'w') as f_handle:
        f_handle.write(json.dumps(scores, indent=4))

if __name__ == '__main__':
    main(sys.argv[1:])
