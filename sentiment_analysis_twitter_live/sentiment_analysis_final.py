#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
stream twitter data with twitter api
"""
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s
from keys import ckey,csecret,atoken,asecret


class Listener(StreamListener):
    f = open('twitter_output.txt','w')
    f.close()
    def on_data(self,data):
        all_data = json.loads(data)
        tweet = all_data["text"]
        sentiment_value,confidence = s.sentiment(tweet)
        if confidence*100 >= 70:
            print tweet,sentiment_value,confidence
            output_file = open('twitter_output.txt','a')
            output_file.write(sentiment_value)
            output_file.write(',')
            output_file.write(str(confidence))
            output_file.write("\n")
            output_file.close()
        
        return True
    
    def on_error(self,status):
        print status
        
auth = OAuthHandler(ckey,csecret)
auth.set_access_token(atoken,asecret)

twitter_stream = Stream(auth,Listener())
twitter_stream.filter(track=['trump'])
