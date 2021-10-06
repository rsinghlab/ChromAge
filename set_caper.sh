#!/bin/bash

conf=".caper/default.conf"

. "$conf"

local-loc-dir="what"

typeset -p local-loc-dir  > "$conf"

