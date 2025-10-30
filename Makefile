NAME		= probit_original
CXX			= c++
CXXFLAGS	= -Wall -Wextra -Werror -std=c++17

SRC_DIR		= src/
OBJ_DIR		= obj/

INCLUDES	= -I
HEADERS		= InverseCumulativeNormal_original.h


SRCS		= main.cpp

OBJS		= $(patsubst $(SRC_DIR)%.cpp,$(OBJ_DIR)%.o,$(SRCS))

all: $(NAME)

$(NAME): $(OBJS) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(NAME)

$(OBJ_DIR)%.o: $(SRC_DIR)%.cpp $(HEADERS)
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR)

fclean: clean
	rm -f $(NAME)

re: fclean all

.PHONY: all, clean, fclean, re
