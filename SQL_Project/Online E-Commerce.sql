CREATE DATABASE ecommerce_db;
USE ecommerce_db;
show tables;

-- Create a database named ecommerce_db?
CREATE DATABASE ecommerce_db;

-- Select the ecommerce_db database?
USE ecommerce_db;

-- Display all databases available in MySQL?
SHOW DATABASES;

 -- Display all tables available in ecommerce_db?  
 SHOW TABLES;
 
 -- Display the structure of the online-ecommerce table?
 DESC `online-ecommerce`;
 
-- Display all records from the online-ecommerce table? 
SELECT * FROM `online-ecommerce`;

-- Display the first 10 records from the table?
SELECT * FROM `online-ecommerce`
LIMIT 10;

-- Display Order Number, Customer Name, Order Date and Status?
SELECT Order_Number, Customer_Name, Order_Date, Status
FROM `online-ecommerce`;

-- Find the total number of orders?
SELECT COUNT(*) AS Total_Orders
FROM `online-ecommerce`;


-- Find the total number of unique customers?
SELECT COUNT(DISTINCT Customer_Name) AS Total_Customers
FROM `online-ecommerce`;


-- Display all unique state codes?
SELECT DISTINCT State_Code
FROM `online-ecommerce`;


-- Display all unique product categories?
SELECT DISTINCT Category
FROM `online-ecommerce`;


-- Display all unique brands?
SELECT DISTINCT Brand
FROM `online-ecommerce`;


-- Display all unique order statuses?
SELECT DISTINCT Status
FROM `online-ecommerce`;


-- Display all delivered orders?
SELECT * FROM `online-ecommerce`
WHERE Status = 'Delivered';


-- Display all processing orders?
SELECT * FROM `online-ecommerce`
WHERE Status = 'Processing';


-- Display all orders from Karnataka (KA)?
SELECT * FROM `online-ecommerce`
WHERE State_Code = 'KA';


-- Display all Samsung products?
SELECT * FROM `online-ecommerce`
WHERE Brand = 'Samsung';


-- Display all products belonging to the CPU category?
SELECT * FROM `online-ecommerce`
WHERE Category = 'CPU';


-- Display orders where quantity is greater than 2?
SELECT Order_Number, Product, Quantity
FROM `online-ecommerce`
WHERE Quantity > 2;


-- Display orders where total sales are greater than 10,000?
SELECT Order_Number, Product, Total_Sales
FROM `online-ecommerce`
WHERE Total_Sales > 10000;


-- Calculate the total quantity sold?
SELECT SUM(Quantity) AS Total_Quantity
FROM `online-ecommerce`;


-- Calculate the total sales?
SELECT SUM(Total_Sales) AS Total_Sales
FROM `online-ecommerce`;


-- Calculate the total cost?
SELECT SUM(Total_Cost) AS Total_Cost
FROM `online-ecommerce`;


-- Calculate the total profit?
SELECT SUM(Total_Sales) - SUM(Total_Cost) AS Total_Profit
FROM `online-ecommerce`;


-- Calculate the average sales per order?
SELECT ROUND(AVG(Total_Sales), 2) AS Average_Sales
FROM `online-ecommerce`;


-- Find the highest sales value?
SELECT MAX(Total_Sales) AS Highest_Sales
FROM `online-ecommerce`;


-- Find the lowest sales value?
SELECT MIN(Total_Sales) AS Lowest_Sales
FROM `online-ecommerce`;

-- Find the number of orders for each status?
SELECT Status, COUNT(*) AS Total_Orders
FROM `online-ecommerce`
GROUP BY Status;


-- Find the total sales for each category?
SELECT Category, SUM(Total_Sales) AS Total_Sales
FROM `online-ecommerce`
GROUP BY Category
ORDER BY Total_Sales DESC;


-- Find the total sales for each brand?
SELECT Brand, SUM(Total_Sales) AS Total_Sales
FROM `online-ecommerce`
GROUP BY Brand
ORDER BY Total_Sales DESC;


-- Find the total sales for each state?
SELECT State_Code, SUM(Total_Sales) AS Total_Sales
FROM `online-ecommerce`
GROUP BY State_Code
ORDER BY Total_Sales DESC;


-- Find the total quantity sold for each category?
SELECT Category, SUM(Quantity) AS Total_Quantity
FROM `online-ecommerce`
GROUP BY Category
ORDER BY Total_Quantity DESC;


-- Find the total profit for each category?
SELECT Category, SUM(Total_Sales) AS Total_Sales, SUM(Total_Cost) AS Total_Cost,
SUM(Total_Sales) - SUM(Total_Cost) AS Profit
FROM `online-ecommerce`
GROUP BY Category
ORDER BY Profit DESC;


-- Find the top 10 customers based on total sales?
SELECT Customer_Name, SUM(Total_Sales) AS Total_Sales
FROM `online-ecommerce`
GROUP BY Customer_Name
ORDER BY Total_Sales DESC
LIMIT 10;


-- Find the top 5 brands based on total profit?
SELECT Brand, SUM(Total_Sales) - SUM(Total_Cost) AS Profit
FROM `online-ecommerce`
GROUP BY Brand
ORDER BY Profit DESC
LIMIT 5;


-- Find the category with the highest total sales?
SELECT Category, SUM(Total_Sales) AS Total_Sales
FROM `online-ecommerce`
GROUP BY Category
ORDER BY Total_Sales DESC
LIMIT 1;


-- Find the state with the highest total sales?
SELECT State_Code, SUM(Total_Sales) AS Total_Sales
FROM `online-ecommerce`
GROUP BY State_Code
ORDER BY Total_Sales DESC
LIMIT 1;


-- Calculate the profit margin for each category?
SELECT 
    Category,
    SUM(Total_Sales) AS Total_Sales,
    SUM(Total_Cost) AS Total_Cost,
    SUM(Total_Sales) - SUM(Total_Cost) AS Profit,
    ROUND(
        (SUM(Total_Sales) - SUM(Total_Cost))
        / SUM(Total_Sales) * 100, 2
    ) AS Profit_Margin
FROM `online-ecommerce`
GROUP BY Category
ORDER BY Profit_Margin DESC;