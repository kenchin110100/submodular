# coding: utf-8
"""
slack-botをメッセージとして送るためのコード
"""
import os

class SlackBot(object):

    @staticmethod
    def post_message(username, text, channel='@k.ikegami', icon_emoji=':thumbsup:',
                     url='https://hooks.slack.com/services/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'):

        cmd = "curl -X POST --data-urlencode " + \
              "\'payload={\"channel\":\"%s\","%channel + \
                         "\"username\":\"%s\","%username + \
                         "\"text\":\"%s\","%text + \
                         "\"icon_emoji\":\"%s\"}\' %s"%(icon_emoji, url)


        os.system(cmd)
