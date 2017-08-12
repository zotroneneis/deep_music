echo "Starting Tensorboard..." &
tensorboard --logdir ~/deepmusic/reports/tensorboard/ --port=6001 &
sleep 2
echo "Opening Tensorboard in Firefox..." &
firefox http://localhost:6001/ &
