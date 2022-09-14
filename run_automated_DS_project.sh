Help()
{
   # Display Help
   echo "Add description of the script functions here."
   echo
   echo "Syntax: scriptTemplate [suffix| -sample-size]"
   echo "options:"
   echo "suffix          Suffix of the experiment. Must be a new suffix."
   echo "sample-size     Subsample the dataset for testing purposes. Not passing this parameter is equivalent to no sampling"
   echo
}

seconds_to_hms()
{
   T=$1
   D=$((T/60/60/24))
   H=$((T/60/60%24))
   M=$((T/60%60))
   S=$((T%60))

   if [ ${D} != 0 ];
   then
      printf '%d days %02d:%02d:%02d' $D $H $M $S
   else
      printf '%02d:%02d:%02d' $H $M $S
   fi
}

run_rapl()
{
# run with RAPL
  echo "starting powerstat" &
  # start up powerstat, redirect output to txt file, prevent output from showing in terminal
  sudo powerstat -DRgf -d=0 1 7200 | sudo tee AutomationOutputs/"$1"/rapl_output_"$1".txt > /dev/null &
  echo "running script with RAPL coverage" &&
  echo "waiting for script to finish" &&
  sudo -E PATH="$PATH" python3 dsc.py --sample="${2:-0}" &&
  echo "script finished, killing processes..."
  sudo pkill -f powerstat ;
  sudo pkill -f dsc.py ;
  echo "all processes killed"
  return 0
}

run_ct()
{
# run with carbontracker
echo "running script with CarbonTracker coverage" &&
sudo -E PATH="$PATH" python3 dsc.py --use_ct --suffix="$1" --sample="${2:-0}" &&  # try without sudo?
return 0
}

install_dependencies()
{
  echo "installing powerstat"
  sudo apt install powerstat
  echo "installing dependencies"
  sudo -E PATH="$PATH" python3 -m pip install -r requirements.txt
}

print_hardware()
{
    sudo lshw -xml | sudo tee AutomationOutputs/"$1"/hardware.xml > /dev/null  # print out hardware info
}

make_directories()
{
  sudo mkdir AutomationOutputs
  sudo mkdir Plots
  sudo mkdir Model_Info
}

run()
{
  install_dependencies
  make_directories
  print_hardware "$1"
  # run with RAPL
  run_rapl "$1" "$2"
  # run with carbontracker
  run_ct "$1" "$2"
  return 0
}

# Get the options, check if -h flag is passed
# Probably can be done more elegantly by assigning options to variables but eh
while getopts ":h" option; do
   case $option in
      h) # display Help
         Help
         exit;;
     \?) # incorrect option
         echo "Error: Invalid option"
         exit;;
   esac
done

if test -f "AutomationOutputs/$1"; then
  echo "suffix $1 already exists! Aborting..."
  return 1
else
  sudo mkdir -p AutomationOutputs/"$1"
  start=$(date +%s)
  run "$1" "$2" &&
  end=$(date +%s)
  duration=$(( end - start ))
  echo
  echo "Finished in "
  seconds_to_hms $duration
fi

return 0
