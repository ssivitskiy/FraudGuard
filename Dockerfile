FROM ubuntu:latest
LABEL authors="stepansivitsky"

ENTRYPOINT ["top", "-b"]