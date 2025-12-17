# Targets
NAME_SINGLECORE		= probit_singlecore
NAME_PARALLELLIZED	= probit_parallel
NAME_VALIDATION		= probit_validation

# Compiler
CXX					= c++

# CXXFLAGS: Standard flags
CXXFLAGS			= -Wall -Wextra -Werror -std=c++17
# OPT_FLAGS: Base flags for all benchmarking runs
OPT_FLAGS			= -O3 -ffast-math -march=native
# VEC_FLAGS: Optimization for parallelization speedup: enable OpenMP
VEC_OPT_FLAGS		= -fopenmp -DENABLE_OMP

SRC_DIR				= srcs/
OBJ_DIR				= obj/

# Includes and headers
INCLUDES			= -I.
HEADERS				= $(SRC_DIR)InverseCumulativeNormal.h

# Sources and objects
SRCS_SINGLECORE		= $(SRC_DIR)benchmark.cpp
SRCS_PARALLELIZED	= $(SRC_DIR)benchmark_paralellized.cpp
SRCS_TESTS			= $(SRC_DIR)tests.cpp

OBJS_SINGLECORE		= $(patsubst $(SRC_DIR)%.cpp,$(OBJ_DIR)%.o,$(SRCS_SINGLECORE))
OBJS_PARALLELIZED	= $(patsubst $(SRC_DIR)%.cpp,$(OBJ_DIR)%.o,$(SRCS_PARALLELIZED))
OBJS_TESTS			= $(patsubst $(SRC_DIR)%.cpp,$(OBJ_DIR)%.o,$(SRCS_TESTS))

# LINKING TARGETS

all: $(NAME_SINGLECORE) $(NAME_PARALLELLIZED) $(NAME_VALIDATION)

# 1. Benchmark baseline, fast core and vector without parallelization
$(NAME_SINGLECORE): $(OBJS_SINGLECORE) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(OPT_FLAGS) $(OBJS_SINGLECORE) -o $(NAME_SINGLECORE)

# 2. Benchmark vector parallelized
$(NAME_PARALLELLIZED): $(OBJS_PARALLELIZED) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(OPT_FLAGS) $(VEC_OPT_FLAGS) $(OBJS_PARALLELIZED) -o $(NAME_PARALLELLIZED)

# 3. Testing Suite Target (tests.cpp only)
$(NAME_VALIDATION): $(OBJS_TESTS) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(OPT_FLAGS) $(OBJS_TESTS) -o $(NAME_VALIDATION)

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
	rm -f $(NAME_SINGLECORE) $(NAME_PARALLELLIZED) $(NAME_VALIDATION)

re: fclean all

.PHONY: all clean fclean re
