#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 22:08:25 2020

@author: t1
"""

import os,time
import numpy as np
import pandas as pd
import json
import torch
import torchtext.data as ttd

def load_spam_data():
    data = pd.read_csv('../data/spam.csv')
    