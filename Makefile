# Targets
NAME_SINGLECORE		= probit_singlecore
NAME_PARALLELLIZED	= probit_parallel
NAME_TESTS			= probit_tests

# Compiler
CXX					= c++

# CXXFLAGS: Standard flags
CXXFLAGS			= -Wall -Wextra -Werror -std=c++17
# OPT_FLAGS: Base flags for all benchmarking runs
OPT_FLAGS			= -O3 -ffast-math -march=native
# VEC_FLAGS: Optimization for parallelization speedup: enable OpenMP
VEC_OPT_FLAGS		= -fopenmp -DENABLE_OMP

SRC_DIR				=
OBJ_DIR				= obj/

# Includes and headers
INCLUDES			= -I.
HEADERS				= InverseCumulativeNormal.h

# Sources and objects
SRCS_SINGLECORE		= benchmark.cpp
SRCS_PARALLELIZED	= benchmark_paralellized.cpp
SRCS_TESTS			= tests.cpp

OBJS_SINGLECORE		= $(patsubst $(SRC_DIR)%.cpp,$(OBJ_DIR)%.o,$(SRCS_SINGLECORE)) #rename this
OBJS_PARALLELIZED	= $(patsubst $(SRC_DIR)%.cpp,$(OBJ_DIR)%.o,$(SRCS_PARALLELIZED))
OBJS_TESTS			= $(patsubst $(SRC_DIR)%.cpp,$(OBJ_DIR)%.o,$(SRCS_TESTS))

# LINKING TARGETS

all: $(NAME_SINGLECORE) $(NAME_PARALLELLIZED) $(NAME_TESTS)

# 1. Benchmark baseline, fast core and vector without parallelization
$(NAME_SINGLECORE): $(OBJS_SINGLECORE) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(OPT_FLAGS) $(OBJS_SINGLECORE) -o $(NAME_SINGLECORE)

# 2. Benchmark vector parallelized
$(NAME_PARALLELLIZED): $(OBJS_PARALLELIZED) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(OPT_FLAGS) $(VEC_OPT_FLAGS) $(OBJS_PARALLELIZED) -o $(NAME_PARALLELLIZED)

# 3. Testing Suite Target (tests.cpp only)
$(NAME_TESTS): $(OBJS_TESTS) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(OPT_FLAGS) $(OBJS_TESTS) -o $(NAME_TESTS)

# COMPILATION OF .o FILES

# Generic rule for MOST .o files (no OpenMP flags)
$(OBJ_DIR)%.o: $(SRC_DIR)%.cpp $(HEADERS)
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(OPT_FLAGS) $(INCLUDES) -c $< -o $@

# Specific rule for PARALLELIZED .o files (which do need OpenMP)
$(OBJS_PARALLELIZED): $(OBJ_DIR)%.o: $(SRC_DIR)%.cpp $(HEADERS)
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(OPT_FLAGS) $(VEC_OPT_FLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR)

fclean: clean
	rm -f $(NAME_SINGLECORE) $(NAME_PARALLELLIZED) $(NAME_TESTS)

re: fclean all

.PHONY: all clean fclean re
