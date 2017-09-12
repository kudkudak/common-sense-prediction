# Common sense prediction project

*Research question*: how far are we from predicting common sense knowledge (from raw text, but not limited to)?

*Extra research question*: how can we further SOTA in word representation learning in terms of representing common
sense knowledge?

We base on http://ttic.uchicago.edu/~kgimpel/papers/li+etal.acl16.pdf.

## Resources

* Write-up: https://www.overleaf.com/10600980csjksczhwpgb

* ACL paper: http://ttic.uchicago.edu/~kgimpel/papers/li+etal.acl16.pdf

* Notes: https://www.evernote.com/Home.action?login=true#b=65f15629-2909-42a0-97c3-0def3cd3e8f5&ses=4&sh=1&sds=5&

## Setup

0. (Optional if you want to run baselines) Clone https://github.com/Lorraine333/ACL_CKBC repository and add to your PYTHONPATH

1. Go to any folder you wish to use as data and run `scripts/fetch_LiACL_data.sh`

2. Configure PYTHONPATH to include root folder of the project. Configure DATA_DIR to point to data directory

## Notes

We use vegab, it is similar to argh, but adds convention that each run necessarily has its own folder, that
after execution will have serialized stdout, stderr, python file and config used.