FROM postgres:15-alpine

RUN apk add --no-cache postgresql-contrib

COPY init-scripts/ /docker-entrypoint-initdb.d/
COPY postgresql.conf /etc/postgresql/postgresql.conf

EXPOSE 5432

USER postgres