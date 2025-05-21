#!/bin/bash

read -p "🔤 Enter a short description for your changes (this will be used as the branch name): " DESCRIPTION
DESCRIPTION=${DESCRIPTION:-default-branch}

BRANCH_NAME=$DESCRIPTION

if [ ! -d ".git" ]; then
  echo "❌ This directory is not a Git repository. Please navigate to your project folder first."
  exit 1
fi

echo "🚧 Starting new GitKraken work item: $BRANCH_NAME"
gk work start "$BRANCH_NAME"

echo "✏️ Please make your code changes now."
read -p "Press Enter when you are ready to commit and push..."
echo "📁 Choose what to stage:"
echo "1) Stage all changes (git add .)"
echo "2) Stage specific files"
read -p "Enter your choice (1 or 2): " stage_choice
if [ "$stage_choice" == "2" ]; then
  read -p "Enter the files to stage (space-separated): " files
  echo "📦 Staging selected files..."
  git add $files
else
  echo "📦 Staging all changes..."
  git add .
fi

echo "🤖 Creating commit with GitKraken AI..."
gk work commit --ai

echo "📤 Pushing changes to remote..."
gk work push

echo "🔀 Creating Pull Request..."
gk work pr create

read -p "🕒 Wait until your PR is merged. When it's done, type 'yes' to run 'gk work end': " confirm_end

if [[ "$confirm_end" == "yes" || "$confirm_end" == "y" ]]; then
  echo "🧹 Finishing up..."
  gk work end
  echo "✅ Work completed and branch cleaned up."
else
  echo "⏳ OK. You can run 'gk work end' manually later."
fi
