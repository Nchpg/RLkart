# Frontend Dockerfile
# nginx-unprivileged runs as the nginx user (101), never root, and listens on
# 8080: a port below 1024 would need a root capability.
FROM nginxinc/nginx-unprivileged:stable-alpine

# Copy static files
COPY ./frontend /usr/share/nginx/html

# Server block only: the image's own nginx.conf already sets the pid and temp
# paths in /tmp, writable by the nginx user.
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 8080
