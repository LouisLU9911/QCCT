## Setup vim theme

1. Open your `.tmux.conf` file (located in your home directory) in Vim by running the following command:

   ```
   vim ~/.tmux.conf
   ```

2. Add the following line to your `.tmux.conf` file to set the TERM variable to xterm-256color:

   ```
   set -g default-terminal "xterm-256color"
   ```

3. Save the changes to your `.tmux.conf` file by typing `:wq` and pressing Enter.

4. Open your `.bashrc` file (located in your home directory) in Vim by running the following command:

   ```
   vim ~/.bashrc
   ```

5. Add the following lines to your `.bashrc` file to set the default iTerm2 terminal colors to match the Iceberg theme:

   ```
   # Set the default iTerm2 terminal colors to match the Iceberg theme
   echo -e "\033]50;SetProfile=Iceberg\a"
   ```

6. Save the changes to your `.bashrc` file by typing `:wq` and pressing Enter.

7. Restart iTerm2 and tmux to see the Iceberg theme applied.

That's it! You should now be able to see the Iceberg theme in both Vim and iTerm2/tmux. Let me know if you have any questions or if there's anything else I can help you with.
