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

run_rapl()
{
# run with RAPL
  sudo mkdir -p AutomationOutputs/rapl_"$1"
  echo "starting powerstat" &
  # start up powerstat, redirect output to txt file, prevent output from showing in terminal
  sudo powerstat -DRgf -d=0 1 500 | sudo tee AutomationOutputs/rapl_"$1"/rapl_output_"$1".txt > /dev/null &
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
sudo -E PATH="$PATH" python3 dsc.py --use_ct --suffix="ct_$1" --sample="${2:-0}" &&
return 0
}

run()
{
  echo "installing powerstat"
  sudo apt install powerstat
  echo "installing dependencies"
  pip3 install -r requirements.txt
  sudo lshw -xml | sudo tee lshw.xml > /dev/null  # print out hardware info
  sudo mkdir AutomationOutputs
  sudo mkdir Plots
  sudo mkdir Model_Info
  # run with RAPL
  run_rapl "$1" "$2"
  # run with carbontracker
  run_ct "$1" "$2"

  return 0
}


# Get the options
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

if test -f "AutomationOutputs/rapl_$1/rapl_output_$1.txt"; then
  echo "suffix $1 already exists"
else
  run "$1" "$2"
fi
