# Image Matcher
A web application written in [Django](https://www.djangoproject.com/) that
leverages the power of ResNet to reverse image search through your local
directory of images.

## Setup
This project recommends settings up with [uv](https://github.com/astral-sh/uv)
but pip may also be used.

Before the application can search through images, it needs to index them. If
you're doing this for the first time with a large of images, this may take a
while.

To begin, create a directory named `photos/` and add your images to it.

#### Then, run the indexer:
```bash
uv run -m modules.index --dataset ./photos/
```
This will create a file called `index.hdf5` at the root on the project. You are
now ready to reverse image search!

If you wish to develop this application please install the pre-commit hook.
You haven't launched the application even once, please manually add the
`pre-commit` by running `uv add pre-commit`.

#### Install pre-commit hook
```bash
uv run pre-commit install
```

## Usage
#### Start Django server:
```bash
uv run manage.py runserver
```
This will start the web app at `127.0.0.1:8000` by default.
