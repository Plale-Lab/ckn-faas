For setup, Please refer to the google doc.

## Note: If you setup your own iluvatar server, replace the ip address in each script's corresponding line.
For example, change 
```
 grpc.aio.insecure_channel("149.165.150.17:8079")
``` 
to
```
 grpc.aio.insecure_channel("xxx.xxx.xxx.xxx:8079")
``` 
8079 is the port that fixed in iluvatar settings.

After the iluvatar is launched, run:

```
/home/exouser/ckn-faas/codespace/code_for_testing/
```
These are the codes that can do the RPC calls, and is the main body of MODE-S.

For example, register_function_rpc.py is the script to register the built functions.


```
python3 register_function_rpc.py
```

The scripts start with "execute" executes the registered functions in different ways. For exmaple, 

```
python3 execute_function_rpc.py
```
Gives a test run of all registered functions.

MODE-S baselines:
+ execute_function_MES_base.py comment/ uncomment line 42-46

MODE-S:
+ execute_MES_async.py
To execute, give a list D (line 15) that has the length of number of incoming requests. Each element in D reflects the deadline in milleseconds.



The scripts start with "quote_est_time", quotes the function estimated time in iluvatar via RPC. To use them, make sure a function with the same name is used.