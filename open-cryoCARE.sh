#!/bin/bash 
echo '[cryoCARE open] loading modules... (imod, motioncor2)'
module load imod/4.9.2
module load motioncor2/1.0.5
echo '[cryoCARE open] starting jupyter notebook (port number may vary!)...'
jupyter notebook --no-browser --port=8889

