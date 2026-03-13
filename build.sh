#!/bin/bash
#
# (C) 2025-2026 Wistron NeWeb Corporation (WNC) - All Rights Reserved
#
# This software and its associated documentation are the confidential and
# proprietary information of Wistron NeWeb Corporation (WNC) ("Company") and
# may not be copied, modified, distributed, or otherwise disclosed to third
# parties without the express written consent of the Company.
#
# Unauthorized reproduction, distribution, or disclosure of this software and
# its associated documentation or the information contained herein is a
# violation of applicable laws and may result in severe legal penalties.
#

# ANSI Color Codes
COLOR_RESET="\033[0m"
COLOR_RED="\033[91m"
COLOR_GREEN="\033[92m"
COLOR_YELLOW="\033[93m"
COLOR_BLUE="\033[94m"
COLOR_BOLD="\033[1m"

echo -e "${COLOR_BOLD}================================================================================ 
                               ${COLOR_BLUE}APP BUILDING STARTED${COLOR_RESET}${COLOR_BOLD}                                
================================================================================${COLOR_RESET}"

cd app

echo -e " ";
echo -e "${COLOR_BOLD}Step 1: Cleaning APP...${COLOR_RESET}"
echo -e "${COLOR_BOLD}--------------------------------------------------------------------------------${COLOR_RESET}"

make clean

echo -e " ";
echo -e "${COLOR_BOLD}Step 2: Building APP...${COLOR_RESET}"
echo -e "${COLOR_BOLD}--------------------------------------------------------------------------------${COLOR_RESET}"

make -j 32

if [ $? -eq 0 ]; then
    echo -e "${COLOR_GREEN}APP build completed successfully.${COLOR_RESET}"

    echo -e " ";
    echo -e "${COLOR_BOLD}--------------------------------------------------------------------------------${COLOR_RESET}"
    # Add message about generated files
    echo -e "${COLOR_BLUE}The following items should now be present in the 'app/build/' directory:${COLOR_RESET}"
    echo -e "  - ${COLOR_YELLOW}wnc-app${COLOR_RESET} (Executable)"
else
    echo -e "${COLOR_RED}APP build failed. Please check the error messages above.${COLOR_RESET}"
    exit 1
fi


echo -e "${COLOR_BOLD}================================================================================ 
                               ${COLOR_GREEN}APP BUILDING FINISHED${COLOR_RESET}${COLOR_BOLD}                                
================================================================================${COLOR_RESET}"

exit 0