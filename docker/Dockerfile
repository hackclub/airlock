FROM kasmweb/ubuntu-noble-desktop:1.18.0

USER root
ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# 0. OPTIMIZATION: Use Hetzner Mirrors (Faster builds on Hetzner servers)
# -----------------------------------------------------------------------------
# This replaces standard Ubuntu mirrors with Hetzner's internal mirror.
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirror.hetzner.com/ubuntu/packages/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|http://mirror.hetzner.com/ubuntu/packages/|g' /etc/apt/sources.list && \
    sed -i 's|http://archive.canonical.com/ubuntu|http://mirror.hetzner.com/ubuntu/packages|g' /etc/apt/sources.list || true

# -----------------------------------------------------------------------------
# 1. CORE SYSTEM UTILITIES & C/C++
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential cmake pkg-config autoconf automake libtool \
    curl wget git unzip zip tar gzip \
    software-properties-common \
    libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev \
    libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev \
    libbz2-dev libexpat1-dev liblzma-dev libffi-dev uuid-dev \
    vim nano htop jq tree \
    ffmpeg imagemagick \
    sqlite3 postgresql-client default-mysql-client \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 2. WINE (Windows .exe Support)
# -----------------------------------------------------------------------------
# 1. Enable 32-bit architecture
RUN dpkg --add-architecture i386

# 2. Install Wine and Winetricks dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wine \
    wine32 \
    wine64 \
    libwine \
    libwine:i386 \
    fonts-wine \
    cabextract \
    zenity \
    && rm -rf /var/lib/apt/lists/*

# 3. Install Winetricks
RUN curl -fsSL https://raw.githubusercontent.com/Winetricks/winetricks/master/src/winetricks -o /usr/local/bin/winetricks && \
    chmod +x /usr/local/bin/winetricks

# -----------------------------------------------------------------------------
# 3. PYTHON (Full Data Science & Web Stack)
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y python3-full python3-pip python3-dev python3-venv

# FIX: Added --ignore-installed to handle 'blinker' and system package conflicts
RUN pip3 install --no-cache-dir --break-system-packages --ignore-installed \
    numpy pandas scipy matplotlib seaborn \
    scikit-learn pillow openpyxl \
    requests flask fastapi uvicorn django \
    boto3 sqlalchemy psycopg2-binary \
    pytest black flake8 ipython jupyterlab \
    beautifulsoup4 lxml pyyaml

# -----------------------------------------------------------------------------
# 4. NODE.JS, BUN, & TYPESCRIPT
# -----------------------------------------------------------------------------
# Install Node 22 (LTS)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs

# FIX: Bun installs directly to BUN_INSTALL, no need to move files
ENV BUN_INSTALL=/usr/local
RUN curl -fsSL https://bun.sh/install | bash

# Install Global NPM Tools
RUN npm install -g npm@latest \
    yarn pnpm \
    typescript ts-node \
    nodemon eslint prettier \
    @angular/cli react-scripts \
    express-generator

# -----------------------------------------------------------------------------
# 5. RUST
# -----------------------------------------------------------------------------
ENV RUSTUP_HOME=/opt/rust
ENV CARGO_HOME=/opt/rust
RUN mkdir -p /opt/rust && chmod -R 777 /opt/rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
ENV PATH="/opt/rust/bin:${PATH}"
RUN cargo install ripgrep bat fd-find

# -----------------------------------------------------------------------------
# 6. GOLANG
# -----------------------------------------------------------------------------
ARG GO_VERSION=1.23.4
RUN rm -rf /usr/local/go && \
    curl -L https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz | tar -C /usr/local -xz
ENV PATH="/usr/local/go/bin:${PATH}"

# -----------------------------------------------------------------------------
# 7. JAVA (OpenJDK 21 + Maven + Gradle)
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y openjdk-21-jdk maven gradle \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 8. .NET 8.0 SDK
# -----------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y dotnet-sdk-8.0 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 9. FINAL CLEANUP & PERMISSIONS
# -----------------------------------------------------------------------------
# Ensure kasm-user owns their home directory
RUN chown -R 1000:1000 /home/kasm-user

# Switch back to Kasm user
USER 1000