FROM mengs-cn-beijing.cr.volces.com/aura/view_eye:basev1

WORKDIR /opt/application
USER root
COPY . .
EXPOSE 8000
# 添加执行权限
RUN chmod +x /opt/application/run.sh
CMD ["bash", "/opt/application/run.sh"]