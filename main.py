def main():
    a = [1, 2, 3]
    b = a          # 不是复制！只是给同一个 list 多贴了个标签
    b.append(4)
    print(a)  # [1, 2, 3, 4]  ← a 也变了！因为 a 和 b 指向同一个东西

if __name__ == "__main__":
    main()
